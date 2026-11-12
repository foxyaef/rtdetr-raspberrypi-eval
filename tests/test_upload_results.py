"""ZIP-only uploads against an in-memory GitHub API; never publish test data."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import unquote
import zipfile

SOURCE=Path(__file__).resolve().parents[1]/'code/upload_results.py'
spec=importlib.util.spec_from_file_location('zip_uploader',SOURCE)
uploader=importlib.util.module_from_spec(spec);spec.loader.exec_module(uploader)


class MemoryAPI:
    def __init__(self):
        self.calls=[];self.refs={'main':'initial'};self.trees={'base':{'README.md':'source'}}
        self.commits={'initial':{'tree':{'sha':'base'}}};self.blobs={};self.race=False;self.can_push=True

    def request(self,method,endpoint,payload=None,*,stream=None,missing_ok=False):
        self.calls.append((method,endpoint,payload))
        path=endpoint.removeprefix('repos/owner/repo/')
        if endpoint=='user':return {'login':'logged-in-user'}
        if endpoint=='repos/owner/repo':return {'permissions':{'push':self.can_push},'default_branch':'main'}
        if method=='GET' and path.startswith('git/ref/heads/'):
            branch=path.removeprefix('git/ref/heads/')
            return {'object':{'sha':self.refs[branch]}} if branch in self.refs else None
        if method=='GET' and path.startswith('contents/'):
            name,head=path.removeprefix('contents/').split('?ref=')
            sha=self.trees[self.commits[head]['tree']['sha']].get(unquote(name))
            return {'sha':sha} if sha else None
        if method=='POST' and path=='git/blobs':
            body=json.load(stream);data=base64.b64decode(body['content'],validate=True)
            sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            self.blobs[sha]=data;return {'sha':sha}
        if method=='GET' and path.startswith('git/commits/'):return self.commits[path.rsplit('/',1)[1]]
        if method=='POST' and path=='git/trees':
            tree=self.trees[payload['base_tree']].copy() if 'base_tree' in payload else {}
            for entry in payload['tree']:tree[entry['path']]=entry['sha']
            name='tree'+str(len(self.trees));self.trees[name]=tree;return {'sha':name}
        if method=='POST' and path=='git/commits':
            name='commit'+str(len(self.commits));self.commits[name]={'tree':{'sha':payload['tree']},'parents':payload['parents']}
            return {'sha':name}
        if method=='POST' and path=='git/refs':
            branch=payload['ref'].removeprefix('refs/heads/')
            if branch in self.refs:raise uploader.APIError('Already exists',422)
            self.refs[branch]=payload['sha'];return {'ref':payload['ref']}
        if method=='PATCH' and path.startswith('git/refs/heads/'):
            assert payload['force'] is False
            branch=path.removeprefix('git/refs/heads/')
            if self.race:
                self.race=False
                self.trees['peer-tree']={**self.trees['base'],'peer.txt':'peer-content'}
                self.commits['peer']={'tree':{'sha':'peer-tree'}};self.refs[branch]='peer'
                raise uploader.APIError('Concurrent update',409)
            if self.commits[payload['sha']]['parents']!=[self.refs[branch]]:raise uploader.APIError('Not fast forward',422)
            self.refs[branch]=payload['sha'];return {'ref':branch}
        raise AssertionError((method,endpoint,payload))


class ZipUploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='zip_upload_')
        self.root=Path(self.temporary.name)
        self.path=self.root/'user1_v1_20261106T011021_267952Z_28425c8c.zip'
        self.requests=self.root/'requests';self.requests.mkdir()
        self.api=MemoryAPI();self.make_zip()

    def tearDown(self):self.temporary.cleanup()

    def make_zip(self,model='user1_v1.onnx',status='complete',extra=None):
        with zipfile.ZipFile(self.path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('provenance.json',json.dumps({'model_name':model,'model_sha256':'fixture'}))
            archive.writestr('evaluation_state.json',json.dumps({'status':status,'stage':'complete'}))
            if status=='complete':archive.writestr('metrics.json','{"image_count":500}')
            archive.writestr('infer.log','Inference 500/500\n')
            for name,data in (extra or {}).items():archive.writestr(name,data)

    def upload(self,**kwargs):
        return uploader.upload_archive(self.path,'owner/repo',api=self.api,temp_dir=self.requests,**kwargs)

    def test_exact_zip_new_branch_source_preserved_temp_request_removed_and_retry_idempotent(self):
        before=self.path.read_bytes();receipt=self.upload()
        self.assertEqual(receipt['branch'],'user1')
        self.assertEqual(receipt['github_login'],'logged-in-user')
        self.assertEqual(self.api.blobs[receipt['git_blob_sha']],before)
        self.assertEqual(self.api.refs['main'],'initial')
        tree=self.api.trees[self.api.commits[receipt['commit_sha']]['tree']['sha']]
        self.assertNotIn('README.md',tree)
        self.assertEqual(self.api.trees['base']['README.md'],'source')
        self.assertEqual(tree[receipt['remote_path']],receipt['git_blob_sha'])
        self.assertEqual(list(self.requests.iterdir()),[])
        self.assertEqual(self.path.read_bytes(),before)
        count=len([call for call in self.api.calls if call[0]=='POST'])
        again=self.upload()
        self.assertTrue(again['already_present'])
        self.assertEqual(again['commit_sha'],receipt['commit_sha'])
        self.assertEqual(len([call for call in self.api.calls if call[0]=='POST']),count)

    def test_check_only_requires_write_permission_and_makes_no_mutations(self):
        result=self.upload(check_only=True)
        self.assertEqual(result['status'],'checked');self.assertFalse(result['published'])
        self.assertTrue(all(call[0]=='GET' for call in self.api.calls))
        self.api.can_push=False
        with self.assertRaisesRegex(uploader.UploadError,'write permission'):self.upload()
        self.assertTrue(all(call[0]=='GET' for call in self.api.calls))

    def test_existing_different_result_is_never_overwritten(self):
        receipt=self.upload();head=self.api.refs['user1']
        self.api.trees[self.api.commits[head]['tree']['sha']][receipt['remote_path']]='different'
        with self.assertRaisesRegex(uploader.UploadError,'different contents'):self.upload()
        self.assertEqual(self.api.refs['user1'],head)

    def test_concurrent_branch_update_keeps_peer_contents(self):
        self.api.refs['user1']='initial';self.api.race=True
        receipt=self.upload()
        self.assertEqual(self.api.commits[receipt['commit_sha']]['parents'],['peer'])
        tree=self.api.trees[self.api.commits[receipt['commit_sha']]['tree']['sha']]
        self.assertEqual(tree['peer.txt'],'peer-content')

    def test_diagnostic_zip_and_free_model_name_are_labeled_without_inference(self):
        self.make_zip(model='인코더 모델.onnx',status='interrupted')
        receipt=self.upload()
        self.assertEqual(receipt['branch'],'shared')
        self.assertEqual(receipt['evaluation_status'],'interrupted')
        self.assertEqual(len(list(self.root.glob('*.onnx'))),0)

    def test_private_paths_and_zip_traversal_are_refused_without_extracting(self):
        for name in ['../outside.json','.env','logs/id_rsa']:
            self.make_zip(extra={name:'do not publish'})
            with self.assertRaises(uploader.UploadError):self.upload()
        self.assertEqual(self.api.calls,[])
        self.assertFalse((self.root.parent/'outside.json').exists())

    def test_insufficient_temp_space_fails_before_blob_creation(self):
        with mock.patch.object(uploader.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(uploader.UploadError,'Temporary request'):self.upload()
        self.assertFalse(any(call[0]=='POST' for call in self.api.calls))

    def test_cli_response_parsing_no_debug_and_token_redaction(self):
        completed=SimpleNamespace(returncode=0,stdout='HTTP/2.0 200 OK\nContent-Type: application/json\n\n{"login":"tester"}\n',stderr='')
        with mock.patch.object(uploader.shutil,'which',return_value='/usr/bin/gh'), \
             mock.patch.object(uploader.subprocess,'run',return_value=completed) as run, \
             mock.patch.dict(uploader.os.environ,{'GH_DEBUG':'api','GH_TOKEN':'ghp_secret123456789'}):
            self.assertEqual(uploader.GitHubCLI().request('GET','user')['login'],'tester')
            self.assertNotIn('GH_DEBUG',run.call_args.kwargs['env'])
            self.assertNotIn('ghp_secret123456789',' '.join(run.call_args.args[0]))
            self.assertNotIn('ghp_secret123456789',uploader.redact('bad ghp_secret123456789'))
            uploader.GitHubCLI().request('GET','repos/owner/repo/contents/results/large.zip?ref=abc')
            self.assertIn('Accept: application/vnd.github.object+json',run.call_args.args[0])
            completed.returncode=1
            completed.stdout='HTTP/2.0 404 Not Found\nContent-Type: application/json\n\n{"message":"Not Found"}'
            self.assertIsNone(uploader.GitHubCLI().request('GET','missing',missing_ok=True))

    def test_default_path_and_receipt_need_only_one_copied_python_file(self):
        info=uploader.inspect_zip(self.path)
        receipt={**info,'status':'uploaded','github_zip_url':'https://example.invalid/result'}
        (self.root/'results').mkdir();default=self.root/'results'/uploader.ZIP_NAME
        default.write_bytes(self.path.read_bytes())
        with mock.patch.object(uploader,'workspace_root',return_value=self.root), \
             mock.patch.object(uploader,'upload_archive',return_value=receipt) as upload:
            self.assertEqual(uploader.main([]),0)
        self.assertEqual(upload.call_args.args[0],default)
        self.assertTrue(Path(str(default)+'.github_upload.json').is_file())
        self.assertEqual(uploader.workspace_root(self.root/'code/upload_results.py'),self.root.resolve())

    def test_copied_single_file_help_needs_no_project_helpers_or_packages(self):
        destination=self.root/'upload_results.py';destination.write_bytes(SOURCE.read_bytes())
        result=subprocess.run([sys.executable,str(destination),'--help'],cwd=self.root,
                              capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--check-only',result.stdout)
        self.assertIn('--zip',result.stdout)


if __name__=='__main__':unittest.main()
