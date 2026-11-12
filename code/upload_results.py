#!/usr/bin/env python3
"""Upload an existing evaluation ZIP using gh login; no evaluation or model needed.

Copy this file alone into code/ or the workspace. Python standard library + gh
are sufficient. No clone, ZIP extraction, pip install, or Git LFS is performed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote
import zipfile

# -------- 기본 설정 --------
ZIP_NAME = "user1_v1_20261106T011021_267952Z_28425c8c.zip"
GITHUB_REPOSITORY = "foxyaef/rtdetr-raspberrypi-eval"
# ---------------------------
MAX_ZIP_BYTES = 90 * 1024 * 1024  # Below GitHub's ordinary 100 MiB Git-file limit.
RESULT_EXTENSIONS = {'.json','.jsonl','.ndjson','.csv','.tsv','.txt','.log','.md','.jpg','.jpeg','.png','.webp','.zip'}
PRIVATE_NAMES = {'.git','.env','.ssh','.netrc','_netrc','.npmrc','.pypirc','credentials','credentials.json','id_rsa','id_ed25519'}


class UploadError(RuntimeError): pass


class APIError(UploadError):
    def __init__(self, message, status=None):
        self.status = status
        super().__init__(message)


def redact(value):
    text = str(value)
    text = re.sub(r'(https?://)[^/\s@]+@',r'\1[redacted]@',text)
    text = re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)\b','[redacted]',text)
    for key in ('GH_TOKEN','GITHUB_TOKEN','GH_ENTERPRISE_TOKEN','GITHUB_ENTERPRISE_TOKEN'):
        secret = os.environ.get(key)
        if secret and len(secret)>7: text=text.replace(secret,'[redacted]')
    return text[-3000:]


def is_link(path):
    return path.is_symlink() or bool(getattr(path,'is_junction',lambda:False)())


def workspace_root(script=None):
    script=Path(script or __file__).resolve()
    return script.parent.parent if script.parent.name=='code' else script.parent


def fingerprint(path):
    size=path.stat().st_size
    sha=hashlib.sha256()
    blob=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            sha.update(block);blob.update(block)
    return {'zip_bytes':size,'zip_sha256':sha.hexdigest(),'git_blob_sha':blob.hexdigest()}


def inspect_zip(path):
    path=Path(path).expanduser().absolute()
    if is_link(path) or not path.is_file():raise UploadError('ZIP must be an existing regular file: '+str(path))
    if path.suffix.lower()!='.zip':raise UploadError('Select a .zip file')
    if path.stat().st_size>MAX_ZIP_BYTES:
        raise UploadError('This ZIP-only uploader accepts up to 90 MiB; larger files need a different storage method.')
    run_id=path.stem
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}',run_id) or '..' in run_id:
        raise UploadError('Keep the evaluator-generated ZIP filename (ASCII run ID).')
    with zipfile.ZipFile(path) as bundle:
        infos=bundle.infolist();names=[item.filename for item in infos]
        if len(names)!=len(set(names)):raise UploadError('Duplicate paths in ZIP')
        if sum(item.file_size for item in infos)>2*1024**3:raise UploadError('Unexpectedly large expanded ZIP contents')
        for item in infos:
            member=PurePosixPath(item.filename)
            if (member.is_absolute() or '..' in member.parts or '\\' in item.filename
                    or any(ord(c)<32 for c in item.filename) or (item.external_attr>>16)&0o170000==0o120000):
                raise UploadError('Unsafe path/link inside ZIP')
            if any(part.lower() in PRIVATE_NAMES or part.lower().startswith('.env') for part in member.parts):
                raise UploadError('Credential/configuration path found inside ZIP; upload refused')
            if not item.is_dir() and member.suffix.lower() not in RESULT_EXTENSIONS:
                raise UploadError('Unexpected result file type: '+item.filename)
        def metadata(name):
            if name not in names:raise UploadError('Missing project metadata: '+name)
            if bundle.getinfo(name).file_size>2*1024*1024:raise UploadError('Oversized metadata: '+name)
            value=json.loads(bundle.read(name).decode('utf-8-sig'))
            if not isinstance(value,dict):raise UploadError('Invalid metadata: '+name)
            return value
        provenance=metadata('provenance.json');state=metadata('evaluation_state.json')
        status=state.get('status')
        if status not in ('complete','interrupted','failed'):raise UploadError('ZIP is not a finished or diagnostic evaluation archive')
        if status=='complete' and 'metrics.json' not in names:raise UploadError('Completed archive is missing metrics.json')
        bad=bundle.testzip()
        if bad:raise UploadError('ZIP CRC check failed: '+bad)
    model_name=provenance.get('model_name','')
    if not isinstance(model_name,str):raise UploadError('Invalid model_name in provenance')
    match=re.match(r'^(user[0-9]+)_',Path(model_name).name)
    branch=match[1] if match else 'shared'
    return {**fingerprint(path),'zip_path':str(path),'run_id':run_id,'branch':branch,
            'model_name':model_name,'evaluation_status':status,
            'remote_path':f'results/{branch}/zip_uploads/{path.name}'}


class GitHubCLI:
    def request(self, method, endpoint, payload=None, *, stream=None, missing_ok=False):
        if not shutil.which('gh'):raise UploadError('gh is not installed. Run: sudo apt install -y gh')
        media='application/vnd.github.object+json' if method=='GET' and '/contents/' in endpoint else 'application/vnd.github+json'
        command=['gh','api','--hostname','github.com','--include','--method',method,
                 '-H','Accept: '+media,endpoint]
        if payload is not None or stream is not None:command+=['--input','-']
        environment=os.environ.copy()
        for name in ('GH_DEBUG','GH_FORCE_TTY','DEBUG','GIT_TRACE','GIT_CURL_VERBOSE','GIT_TRACE_CURL'):
            environment.pop(name,None)
        environment.update(GH_PROMPT_DISABLED='1',NO_COLOR='1')
        kwargs={'capture_output':True,'text':True,'encoding':'utf-8','errors':'replace',
                'timeout':1800,'env':environment}
        if stream is not None:kwargs['stdin']=stream
        elif payload is not None:kwargs['input']=json.dumps(payload,ensure_ascii=False)
        result=subprocess.run(command,**kwargs)
        headers,separator,body=result.stdout.partition('\n\n')
        match=re.match(r'HTTP/\S+\s+(\d+)',headers)
        status=int(match[1]) if match else None
        if status==404 and missing_ok:return None
        if result.returncode or status is not None and status>=400:
            raise APIError(f'GitHub request failed ({method} {endpoint}, HTTP {status}): '+redact(body or result.stderr),status)
        try:return json.loads(body if separator else result.stdout)
        except ValueError as error:raise APIError('GitHub returned an unexpected response',status) from error


def upload_blob(api, repo, path, info, temp_dir=None):
    # Base64 JSON is streamed into a temporary request file, never a full Python
    # string. Usually /tmp (RAM-backed on this Pi). It is removed automatically.
    directory=Path(temp_dir or tempfile.gettempdir())
    required=((info['zip_bytes']+2)//3)*4+1024*1024
    if not directory.is_dir() or shutil.disk_usage(directory).free<required:
        raise UploadError(f'Temporary request needs approximately {required/1048576:.1f} MiB free at {directory}')
    with tempfile.TemporaryFile(mode='w+b',dir=directory) as request:
        request.write(b'{"encoding":"base64","content":"')
        with path.open('rb') as source:
            for block in iter(lambda:source.read(3*256*1024),b''):request.write(base64.b64encode(block))
        request.write(b'"}');request.flush();request.seek(0)
        if fingerprint(path)['zip_sha256']!=info['zip_sha256']:raise UploadError('ZIP changed while preparing upload')
        result=api.request('POST',f'repos/{repo}/git/blobs',stream=request)
    if result.get('sha')!=info['git_blob_sha']:raise UploadError('Uploaded blob SHA does not match the selected ZIP')
    return result['sha']


def upload_archive(path, repo=GITHUB_REPOSITORY, *, check_only=False, temp_dir=None, api=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*',repo) or '..' in repo:
        raise UploadError('Repository must be owner/repository, without a URL or credentials')
    print('1/4 ZIP validation:',path,flush=True)
    info=inspect_zip(path);path=Path(info['zip_path'])
    print('Evaluation:',info['evaluation_status'],'ZIP MiB:',round(info['zip_bytes']/1048576,2),flush=True)
    api=api or GitHubCLI()
    user=api.request('GET','user')
    repository=api.request('GET',f'repos/{repo}')
    if not repository.get('permissions',{}).get('push'):
        raise UploadError('The logged-in GitHub account lacks write permission for this repository')
    info.update(repository=repo,github_login=user['login'])
    print('2/4 Authenticated:',user['login'],'Target branch:',info['branch'],flush=True)
    if check_only:return {**info,'status':'checked','published':False}
    branch=info['branch'];blob_sha=None
    for attempt in range(3):
        reference=api.request('GET',f'repos/{repo}/git/ref/heads/{branch}',missing_ok=True)
        new_branch=reference is None
        if new_branch:
            reference=api.request('GET',f'repos/{repo}/git/ref/heads/{quote(repository["default_branch"],safe="/")}')
        head=reference['object']['sha']
        existing=api.request('GET',f'repos/{repo}/contents/{quote(info["remote_path"],safe="/")}?ref={head}',missing_ok=True)
        if existing is not None:
            if not isinstance(existing,dict) or existing.get('sha')!=info['git_blob_sha']:
                raise UploadError('This result path already exists with different contents; nothing was overwritten')
            commit=head;already_present=True;break
        if blob_sha is None:
            print('3/4 Uploading ZIP only; no clone, extraction or model upload.',flush=True)
            blob_sha=upload_blob(api,repo,path,info,temp_dir)
        tree_payload={'tree':[{'path':info['remote_path'],'mode':'100644','type':'blob','sha':blob_sha}]}
        if not new_branch:
            base=api.request('GET',f'repos/{repo}/git/commits/{head}')
            tree_payload['base_tree']=base['tree']['sha']
        tree=api.request('POST',f'repos/{repo}/git/trees',tree_payload)
        commit=api.request('POST',f'repos/{repo}/git/commits',{
            'message':f'Upload result ZIP {info["run_id"]} ({info["evaluation_status"]})',
            'tree':tree['sha'],'parents':[head]})['sha']
        if fingerprint(path)['zip_sha256']!=info['zip_sha256']:raise UploadError('ZIP changed before publishing')
        try:
            if new_branch:api.request('POST',f'repos/{repo}/git/refs',{'ref':'refs/heads/'+branch,'sha':commit})
            else:api.request('PATCH',f'repos/{repo}/git/refs/heads/{branch}',{'sha':commit,'force':False})
            already_present=False;break
        except APIError as error:
            if error.status not in (409,422) or attempt==2:raise
            print('Branch changed or rejected; checking again without force.',flush=True)
    url=f'https://github.com/{repo}/blob/{commit}/{quote(info["remote_path"],safe="/")}'
    print('4/4 ZIP upload complete:',url,flush=True)
    return {**info,'status':'uploaded','published':True,'commit_sha':commit,'already_present':already_present,
            'github_zip_url':url,'github_branch_url':f'https://github.com/{repo}/tree/{branch}'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,help='Existing evaluation ZIP; relative paths are relative to this script workspace')
    parser.add_argument('--repo',default=GITHUB_REPOSITORY)
    parser.add_argument('--check-only',action='store_true',help='Validate ZIP, login and repository permissions; no upload')
    parser.add_argument('--temp-dir',type=Path,help='Temporary API request directory; default system /tmp')
    args=parser.parse_args(argv)
    root=workspace_root()
    path=args.zip or Path('results')/ZIP_NAME
    if not path.is_absolute():path=root/path
    try:
        receipt=upload_archive(path,args.repo,check_only=args.check_only,temp_dir=args.temp_dir)
        if not args.check_only:
            destination=Path(str(path)+'.github_upload.json')
            try:
                if is_link(destination):raise OSError('Receipt path cannot be a symlink')
                destination.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                print('Receipt:',destination,flush=True)
            except OSError as error:print('Upload succeeded, but local receipt could not be saved:',redact(error),flush=True)
        print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
        return 0
    except (UploadError,OSError,ValueError,zipfile.BadZipFile,subprocess.SubprocessError) as error:
        print('Upload failed; original ZIP preserved:',redact(error),file=sys.stderr,flush=True)
        print('Check login with: gh auth status --hostname github.com',file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
