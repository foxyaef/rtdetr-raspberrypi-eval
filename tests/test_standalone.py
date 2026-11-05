"""One-file bootstrap safety and isolated-workspace behavior.

The optional COCO_DATA_DIR integration is read-only and uses existing locked
images; no test downloads images or posts a model/result to GitHub.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zlib

ROOT=Path(__file__).resolve().parents[1]
LAUNCHER=ROOT/'code/test.py'


def load_launcher():
    spec=importlib.util.spec_from_file_location('standalone_launcher_under_test',LAUNCHER)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandaloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher=load_launcher()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='one_file_')
        self.folder=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_workspace_location_for_loose_and_code_folder_launchers(self):
        loose=self.folder/'SSH 계정 1'/'test.py'
        organized=self.folder/'SSH 계정 2'/'code'/'test.py'
        self.assertEqual(self.launcher.workspace_root(loose),loose.parent.resolve())
        self.assertEqual(self.launcher.workspace_root(organized),organized.parent.parent.resolve())

    def test_help_works_from_only_copied_file_without_helpers_or_install(self):
        home=self.folder/'SSH 계정 with spaces'
        home.mkdir()
        script=home/'test.py';shutil.copyfile(LAUNCHER,script)
        result=subprocess.run([sys.executable,str(script),'--help'],cwd=self.folder,
                              capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--setup-only',result.stdout)
        self.assertIn('--use-current-python',result.stdout)
        self.assertFalse((home/'.rtdetr_venv').exists())
        self.assertFalse((home/'.rtdetr_runtime').exists())
        self.assertEqual({p.name for p in home.iterdir()},{'test.py'})

    def test_materializing_bundle_preserves_model_and_result_files(self):
        home=self.folder/'isolated user'
        (home/'models').mkdir(parents=True);(home/'results').mkdir()
        model=home/'models/user1_existing.onnx';model.write_bytes(b'user model sentinel')
        saved=home/'results/old.json';saved.write_text('{"unchanged":true}',encoding='utf-8')
        runtime=self.launcher.materialize_bundle(home)
        self.assertTrue((runtime/'code/benchmark.py').is_file())
        self.assertTrue((runtime/'code/coco_data.py').is_file())
        self.assertTrue((runtime/'code/coco_metrics.py').is_file())
        self.assertTrue((runtime/'code/github_upload.py').is_file())
        self.assertTrue((runtime/'protocol.json').is_file())
        self.assertTrue((runtime/'requirements.txt').is_file())
        self.assertEqual(model.read_bytes(),b'user model sentinel')
        self.assertEqual(saved.read_text(encoding='utf-8'),'{"unchanged":true}')
        self.assertEqual(self.launcher.materialize_bundle(home),runtime)

    def test_corrupt_embedded_payload_is_rejected_before_file_extraction(self):
        home=self.folder/'home'
        with mock.patch.object(self.launcher,'BUNDLE_DATA','not a valid payload!'):
            with self.assertRaises((ValueError,RuntimeError)):
                self.launcher.materialize_bundle(home)
        self.assertFalse((home/'code/benchmark.py').exists())
        self.assertFalse((home/'models/user1_v1.onnx').exists())

    def test_changed_runtime_cannot_be_silently_reused_as_trusted_code(self):
        home=self.folder/'home';home.mkdir();runtime=self.launcher.materialize_bundle(home)
        engine=runtime/'code/benchmark.py';expected=engine.read_bytes()
        engine.write_bytes(b'raise RuntimeError("tampered cached code")\n')
        try:
            again=self.launcher.materialize_bundle(home)
        except (ValueError,RuntimeError):
            return  # Refusing a changed runtime also prevents executing it.
        self.assertEqual((again/'code/benchmark.py').read_bytes(),expected)

    def test_current_python_option_does_not_install_missing_dependencies(self):
        home=self.folder/'home';home.mkdir();runtime=self.launcher.materialize_bundle(home)
        with mock.patch.object(self.launcher,'environment_versions',return_value={}), \
             mock.patch.object(self.launcher.subprocess,'run') as run:
            with self.assertRaises((ValueError,RuntimeError)):
                self.launcher.ensure_environment(home,runtime,use_current=True)
        run.assert_not_called()

    def test_valid_outer_hash_cannot_authorize_unexpected_or_traversal_paths(self):
        original=self.launcher.decode_bundle()
        for path in ['../escape.py','/escape.py','C:\\escape.py']:
            with self.subTest(path=path):
                crafted=copy.deepcopy(original)
                crafted['files'][path]=crafted['files'].pop('code/coco_metrics.py')
                raw=json.dumps(crafted).encode('utf-8')
                encoded=base64.b64encode(zlib.compress(raw)).decode('ascii')
                home=self.folder/('path_'+str(len(path)));home.mkdir(exist_ok=True)
                with mock.patch.object(self.launcher,'BUNDLE_DATA',encoded), \
                     mock.patch.object(self.launcher,'BUNDLE_SHA256',hashlib.sha256(raw).hexdigest()):
                    with self.assertRaises(RuntimeError):self.launcher.materialize_bundle(home)
                self.assertFalse((home/'.rtdetr_runtime').exists())
        self.assertFalse((self.folder/'escape.py').exists())

    def test_individual_file_hash_must_match_even_when_bundle_hash_is_valid(self):
        crafted=copy.deepcopy(self.launcher.decode_bundle())
        crafted['files']['code/benchmark.py']['content']+='\nraise RuntimeError("unexpected change")\n'
        raw=json.dumps(crafted).encode('utf-8')
        with mock.patch.object(self.launcher,'BUNDLE_DATA',base64.b64encode(zlib.compress(raw)).decode('ascii')), \
             mock.patch.object(self.launcher,'BUNDLE_SHA256',hashlib.sha256(raw).hexdigest()):
            with self.assertRaises(RuntimeError):self.launcher.decode_bundle()

    def test_unowned_private_environment_is_preserved_and_never_modified(self):
        home=self.folder/'home';home.mkdir();runtime=self.launcher.materialize_bundle(home)
        directory=home/'.rtdetr_venv';directory.mkdir()
        sentinel=directory/'unrelated-user-file.txt';sentinel.write_text('preserve',encoding='utf-8')
        with mock.patch.object(self.launcher.subprocess,'run') as run:
            with self.assertRaises(RuntimeError):self.launcher.ensure_environment(home,runtime)
        run.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding='utf-8'),'preserve')
        self.assertFalse((directory/self.launcher.ENVIRONMENT_MARKER).exists())

    def test_setup_lock_does_not_remove_another_active_owner(self):
        home=self.folder/'home';home.mkdir()
        with self.launcher.setup_lock(home):
            contents=(home/'.rtdetr_setup.lock').read_bytes()
            with self.assertRaises(RuntimeError):
                with self.launcher.setup_lock(home):pass
            self.assertEqual((home/'.rtdetr_setup.lock').read_bytes(),contents)
        self.assertFalse((home/'.rtdetr_setup.lock').exists())

    def test_launch_preserves_unicode_arguments_and_uses_workspace_environment(self):
        home=self.folder/'SSH 계정 user 2';home.mkdir()
        data=self.folder/'공통 이미지 with spaces'
        with mock.patch.object(self.launcher,'workspace_root',return_value=home), \
             mock.patch.object(self.launcher,'ensure_environment',return_value=Path(sys.executable)), \
             mock.patch.object(self.launcher.subprocess,'call',return_value=7) as call, \
             mock.patch.object(self.launcher,'MODEL_NAME','user2_encoder_v3.onnx'), \
             mock.patch.object(self.launcher,'UPLOAD_TO_GITHUB',False):
            code=self.launcher.launch(['--setup-only','--use-current-python','--data-dir',str(data),'--no-download'])
        self.assertEqual(code,7)
        args=call.call_args.args[0];kwargs=call.call_args.kwargs
        self.assertIsInstance(args,list)
        self.assertEqual(args[0],str(Path(sys.executable)))
        self.assertIn('--prepare-only',args)
        self.assertNotIn('--setup-only',args)
        self.assertNotIn('--use-current-python',args)
        self.assertEqual(args[args.index('--data-dir')+1],str(data))
        self.assertEqual(kwargs['env']['RTDETR_WORKSPACE_ROOT'],str(home))
        self.assertEqual(kwargs['env']['RTDETR_MODEL_NAME'],'user2_encoder_v3.onnx')
        self.assertEqual(kwargs['env']['RTDETR_UPLOAD'],'0')
        self.assertNotIn('PYTHONPATH',kwargs['env'])
        self.assertFalse(kwargs.get('shell',False))
        for name in ['code','models','data','results']:self.assertTrue((home/name).is_dir())

    def test_pip_bootstrap_rejects_cross_host_redirect_and_preserves_destination(self):
        destination=self.folder/'get-pip.py'
        response=mock.MagicMock()
        response.__enter__.return_value=response
        response.geturl.return_value='https://other.example/get-pip.py'
        with mock.patch.object(self.launcher.urllib.request,'urlopen',return_value=response):
            with self.assertRaises(RuntimeError):self.launcher._download_get_pip(destination)
        self.assertFalse(destination.exists())

    def test_environment_that_reports_host_prefix_is_rejected_before_pip(self):
        home=self.folder/'home';home.mkdir();runtime=self.launcher.materialize_bundle(home)
        directory=home/'.rtdetr_venv';directory.mkdir()
        (directory/self.launcher.ENVIRONMENT_MARKER).write_text(json.dumps({'format':1,'purpose':'rtdetr-standalone'}),encoding='utf-8')
        python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        python.parent.mkdir();python.write_bytes(b'placeholder')
        host_info={'prefix':sys.prefix,'base_prefix':sys.base_prefix,'bits':64,'version':[3,12]}
        with mock.patch.object(self.launcher.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(host_info),stderr='')) as run, \
             mock.patch.object(self.launcher,'environment_versions') as versions:
            with self.assertRaises(RuntimeError):self.launcher.ensure_environment(home,runtime)
        self.assertEqual(run.call_count,1)
        self.assertEqual(run.call_args.args[0][0],str(python))
        self.assertEqual(run.call_args.args[0][1],'-c')
        versions.assert_not_called()

    def test_missing_pip_fallback_installs_only_inside_the_private_environment(self):
        home=self.folder/'SSH 사용자 with spaces';home.mkdir();runtime=self.launcher.materialize_bundle(home)
        directory=home/'.rtdetr_venv'
        python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        expected=self.launcher.parse_requirements(runtime/'requirements.txt')
        missing={name:None for name in expected}
        commands=[]
        def create(target):
            self.assertEqual(Path(target),directory)
            python.parent.mkdir(parents=True,exist_ok=True);python.write_bytes(b'placeholder')
        def run(args,**kwargs):
            commands.append((args,kwargs))
            self.assertIsInstance(args,list)
            self.assertEqual(args[0],str(python))
            self.assertFalse(kwargs.get('shell',False))
            self.assertEqual(kwargs['env']['PYTHONNOUSERSITE'],'1')
            self.assertNotIn('--user',args)
            if args[1]=='-c':
                info={'prefix':str(directory),'base_prefix':str(self.folder/'host-python'),
                      'bits':64,'version':[3,12]}
                return SimpleNamespace(returncode=0,stdout=json.dumps(info),stderr='')
            if args[1:4]==['-m','pip','--version'] or args[1:3]==['-m','ensurepip']:
                return SimpleNamespace(returncode=1,stdout='',stderr='not installed')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        def download(path):
            path.write_text('# official-bootstrap stand-in for offline test\n',encoding='utf-8')
            return {'url':self.launcher.GET_PIP_URL,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        with mock.patch.object(self.launcher.venv,'EnvBuilder',return_value=SimpleNamespace(create=create)) as builder, \
             mock.patch.object(self.launcher,'environment_versions',side_effect=[missing,expected]), \
             mock.patch.object(self.launcher,'_download_get_pip',side_effect=download) as fetch, \
             mock.patch.object(self.launcher.subprocess,'run',side_effect=run):
            selected=self.launcher.ensure_environment(home,runtime)
        self.assertEqual(selected,python)
        self.assertTrue(builder.called)
        self.assertFalse(builder.call_args.kwargs.get('clear',False))
        fetch.assert_called_once_with(runtime/'get-pip.py')
        installs=[args for args,_ in commands if args[1:4]==['-m','pip','install']]
        self.assertEqual(len(installs),1)
        self.assertEqual(installs[0][installs[0].index('-r')+1],str(runtime/'requirements.txt'))
        self.assertIn([str(python),str(runtime/'get-pip.py'),'--index-url','https://pypi.org/simple'],[args for args,_ in commands])
        provenance=json.loads((runtime/'installer_provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['python'],str(python))
        self.assertFalse(provenance['system_python_modified'])

    def test_installer_ignores_ambient_global_pip_targets_and_user_sites(self):
        dirty={'PIP_INDEX_URL':'https://untrusted.invalid/simple','PIP_EXTRA_INDEX_URL':'https://extra.invalid/simple',
               'PIP_TARGET':'/unrelated/global/packages','PIP_USER':'true','PIP_TRUSTED_HOST':'untrusted.invalid',
               'PYTHONHOME':'/unrelated/python','PYTHONPATH':'/unrelated/modules','PYTHONUSERBASE':'/unrelated/usersite',
               'HTTPS_PROXY':'http://proxy.example:8080'}
        with mock.patch.dict(os.environ,dirty):
            env=self.launcher._child_environment()
        for key in dirty:
            if key!='HTTPS_PROXY':self.assertNotIn(key,env)
        self.assertEqual(env['HTTPS_PROXY'],dirty['HTTPS_PROXY'])
        self.assertEqual(env['PIP_CONFIG_FILE'],os.devnull)
        self.assertEqual(env['PIP_REQUIRE_VIRTUALENV'],'true')
        self.assertEqual(env['PYTHONNOUSERSITE'],'1')

    @unittest.skipUnless(os.environ.get('COCO_DATA_DIR'),'Set COCO_DATA_DIR for copied-file offline integration')
    def test_copied_file_prepares_only_using_pinned_current_python_and_existing_images(self):
        home=self.folder/'standalone SSH 사용자'
        home.mkdir();script=home/'test.py';shutil.copyfile(LAUNCHER,script)
        result=subprocess.run([sys.executable,str(script),'--use-current-python','--setup-only',
                               '--no-download','--data-dir',os.environ['COCO_DATA_DIR']],
                              cwd=self.folder,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        for name in ['code','models','data','results']:
            self.assertTrue((home/name).is_dir(),name)
        self.assertTrue(list((home/'.rtdetr_runtime').glob('*/code/benchmark.py')))
        self.assertFalse((home/'.rtdetr_venv').exists())
        self.assertFalse((home/'data/coco/downloads').exists())


if __name__=='__main__':unittest.main()
