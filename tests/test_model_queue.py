"""Sequential scheduling, durable resume, and copied-file ONNX queue integration."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import model_queue as queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='queue_')
        self.root=Path(self.temp.name)
        self.inbox=self.root/'queue/inbox'; self.inbox.mkdir(parents=True)
        self.calls=[]; self.prepared=object()

    def tearDown(self): self.temp.cleanup()

    def add(self,name,content=b'model fixture'):
        (self.inbox/name).write_bytes(content)

    def evaluate(self,name,digest,prepared):
        self.assertIs(prepared,self.prepared)
        self.assertEqual(digest,queue._sha256(self.inbox/name))
        self.calls.append(name)
        output=self.root/'results'/(Path(name).stem+f'_{len(self.calls)}')
        output.mkdir(parents=True)
        archive=Path(str(output)+'.zip');archive.write_bytes(b'fixture archive')
        return {'returncode':0,'result_dir':str(output),'archive_path':str(archive),'upload_status':'disabled'}

    def run_queue(self,**kwargs):
        prepare=mock.Mock(return_value=self.prepared)
        result=queue.run_queue(self.root,{'limit':3},prepare,kwargs.pop('evaluate',self.evaluate),**kwargs)
        return result,prepare

    def state(self): return json.loads((self.root/'queue/state.json').read_text(encoding='utf-8'))

    def test_failure_continues_preparation_once_and_resume_skips_completed(self):
        for name in ['user3_v1.onix','user2_v1.onnx','user1_v1.onnx']:self.add(name)
        def evaluate(name,*args):
            if name.startswith('user2'): self.calls.append(name); raise RuntimeError('broken ONNX')
            result=self.evaluate(name,*args)
            if name.startswith('user3'):result.update(returncode=2,upload_status='failed')
            return result
        result,prepare=self.run_queue(evaluate=evaluate)
        self.assertEqual(result,1);prepare.assert_called_once_with()
        self.assertEqual(self.calls,['user1_v1.onnx','user2_v1.onnx','user3_v1.onix'])
        self.assertEqual([j['status'] for j in self.state()['jobs']],['completed','failed','completed'])
        _,prepare=self.run_queue()
        prepare.assert_not_called();self.assertEqual(len(self.calls),3)
        result,_=self.run_queue(retry_failed=True)
        self.assertEqual(result,0);self.assertEqual(self.calls[-1],'user2_v1.onnx')
        self.assertEqual(self.state()['jobs'][1]['attempts'],2)

    def test_changed_file_is_new_job_and_partial_upload_is_ignored(self):
        self.add('user1_v1.onnx');self.add('user2_v1.onnx.part')
        self.run_queue();self.add('user1_v1.onnx',b'changed')
        self.run_queue()
        self.assertEqual(self.calls,['user1_v1.onnx','user1_v1.onnx'])
        self.assertEqual(len(self.state()['jobs']),2)

    def test_explicit_order_invalid_name_rejection_and_deduplication(self):
        self.add('user1_v1.onnx');self.add('user2_v1.onix');self.add('user8_v1.onnx')
        self.run_queue(names=['user2_v1.onix','user1_v1.onnx','user2_v1.onix','../escape.onnx'])
        self.assertEqual(self.calls,['user2_v1.onix','user1_v1.onnx'])
        self.assertEqual(len(self.state()['rejected']),1)

    def test_interrupted_job_restarts_without_retry_failed_flag(self):
        self.add('user1_v1.onnx')
        self.assertEqual(self.run_queue(evaluate=mock.Mock(side_effect=KeyboardInterrupt))[0],130)
        self.assertEqual(self.state()['jobs'][0]['status'],'interrupted')
        self.assertEqual(self.run_queue()[0],0)
        self.assertEqual(self.state()['jobs'][0]['attempts'],2)

    def test_watch_processes_new_model_without_looping_failed_jobs(self):
        self.add('user1_v1.onnx')
        def tick(seconds):
            if not (self.inbox/'user2_v1.onix').exists():self.add('user2_v1.onix')
            else:raise KeyboardInterrupt
        with mock.patch.object(queue.time,'sleep',side_effect=tick):
            self.assertEqual(self.run_queue(watch=True,poll_seconds=2)[0],130)
        self.assertEqual(self.calls,['user1_v1.onnx','user2_v1.onix'])

    def test_os_lock_blocks_another_process_and_releases(self):
        command=[sys.executable,'-c',
            'import sys;sys.path.insert(0,sys.argv[1]);from model_queue import evaluation_lock;'
            'lock=evaluation_lock(sys.argv[2]);lock.__enter__();lock.__exit__(None,None,None)',
            str(ROOT/'code'),str(self.root)]
        with queue.evaluation_lock(self.root):
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('already active',result.stderr)
        self.assertEqual(subprocess.run(command,capture_output=True,timeout=30).returncode,0)

    def test_result_archive_is_stable_after_upload_receipt(self):
        spec=importlib.util.spec_from_file_location('queue_benchmark',ROOT/'code/benchmark.py')
        engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        output=self.root/'run_001';output.mkdir();(output/'metrics.json').write_text('{}')
        archive=Path(engine.package_results(output));digest=queue._sha256(archive)
        (output/'upload_status.json').write_text('{"status":"uploaded"}')
        engine.package_results(output)
        self.assertEqual(queue._sha256(archive),digest)
        with zipfile.ZipFile(archive) as bundle:self.assertEqual(bundle.namelist(),['metrics.json'])

    @unittest.skipUnless(os.environ.get('COCO_DATA_DIR'),'Set COCO_DATA_DIR for real copied-file queue integration')
    def test_copied_file_auto_runs_two_real_onnx_models_around_a_failure(self):
        import onnx
        from onnx import helper,TensorProto
        script=self.root/'test.py';shutil.copyfile(ROOT/'code/test.py',script)
        inputs=[helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,640,640]),
                helper.make_tensor_value_info('orig_target_sizes',TensorProto.INT64,[1,2])]
        definitions=[('labels',TensorProto.INT64,[1,1],[0]),
                     ('boxes',TensorProto.FLOAT,[1,1,4],[1,1,10,10]),
                     ('scores',TensorProto.FLOAT,[1,1],[.5])]
        nodes=[helper.make_node('Constant',[],[name],value=helper.make_tensor(name+'_value',kind,shape,values))
               for name,kind,shape,values in definitions]
        outputs=[helper.make_tensor_value_info(name,kind,shape) for name,kind,shape,_ in definitions]
        model=helper.make_model(helper.make_graph(nodes,'queue_control_fixture',inputs,outputs),
                                opset_imports=[helper.make_opsetid('',17)])
        model.ir_version=10
        for name in ['user1_fixture.onnx','user3_fixture.onix']:onnx.save(model,self.inbox/name)
        self.add('user2_broken.onnx',b'not an ONNX model')
        command=[sys.executable,str(script),'--use-current-python','--no-download','--no-upload',
                 '--data-dir',os.environ['COCO_DATA_DIR'],'--limit','2','--threads','1']
        result=subprocess.run(command,cwd=self.root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
        self.assertEqual(result.returncode,1,result.stdout+'\n'+result.stderr)
        jobs=self.state()['jobs']
        self.assertEqual([j['status'] for j in jobs],['completed','failed','completed'],result.stdout+result.stderr)
        for job in jobs:
            self.assertTrue(Path(job['archive_path']).is_file())
            with zipfile.ZipFile(job['archive_path']) as bundle:
                self.assertIn('provenance.json',bundle.namelist())
                self.assertIn('evaluation_state.json',bundle.namelist())
        for job in [jobs[0],jobs[2]]:
            metrics=json.loads((Path(job['result_dir'])/'metrics.json').read_text())
            self.assertEqual(metrics['image_count'],2)
            self.assertFalse(metrics['full_evaluation'])
            self.assertGreater(metrics['resource']['rss_peak_bytes'],0)
        again=subprocess.run(command,cwd=self.root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
        self.assertEqual([j['attempts'] for j in self.state()['jobs']],[1,1,1])
        self.assertNotIn('Stage: infer',again.stdout)


if __name__=='__main__':unittest.main()
