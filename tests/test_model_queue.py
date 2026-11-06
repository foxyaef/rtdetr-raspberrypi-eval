"""Sequential scheduling, durable resume, and copied-file ONNX queue integration."""
import importlib.util
import errno
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
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

    def evaluate(self,name,digest,prepared,progress):
        self.assertIs(prepared,self.prepared)
        self.assertEqual(digest,queue._sha256(self.inbox/name))
        self.calls.append(name)
        progress({'stage':'infer','image_current':1,'image_total':3})
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
        self.add('user1_v1.onnx');self.add('user2_v1.onix');self.add('.hidden.onnx')
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

    def test_default_cli_watches_empty_inbox_and_once_is_explicit(self):
        spec=importlib.util.spec_from_file_location('queue_cli_benchmark',ROOT/'code/benchmark.py')
        engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        with mock.patch.object(engine,'ROOT',self.root), mock.patch.object(queue,'run_queue',return_value=0) as run:
            self.assertEqual(engine.main(['--no-upload','--no-download']),0)
            self.assertTrue(run.call_args.kwargs['watch'])
            self.assertEqual(run.call_args.args[1]['limit'],500)
            self.assertEqual(engine.main(['--once','--no-upload','--no-download']),0)
            self.assertFalse(run.call_args.kwargs['watch'])
        with self.assertRaises(SystemExit),mock.patch.object(engine,'ROOT',self.root):
            engine.main(['--once','--watch','--no-download'])

    def test_txt_heartbeat_detects_arrivals_while_model_is_busy(self):
        self.add('user1_v1.onnx')
        original=queue.QueueStatus
        original_sleep=time.sleep
        def status(*args,**kwargs):return original(*args,**kwargs,interval=.02)
        def sleep(seconds):
            if seconds==15:raise KeyboardInterrupt
            original_sleep(seconds)
        def evaluate(name,digest,prepared,progress):
            progress({'stage':'infer','image_current':25,'image_total':1000})
            if name=='user1_v1.onnx':
                self.add('user2_v1.onix')
                deadline=time.monotonic()+3
                while time.monotonic()<deadline:
                    text=(self.root/'queue/status.txt').read_text(encoding='utf-8')
                    if '대기 중인 모델: 1건' in text and 'user2_v1.onix' in text:break
                    time.sleep(.01)
                else:self.fail('New arrival was not displayed while inference was active')
                self.assertIn('실행 중인 모델: user1_v1.onnx',text)
                self.assertIn('추론 진행: 25/1000장',text)
                self.assertNotIn('바로 사용 가능',text)
            return self.evaluate(name,digest,prepared,progress)
        with mock.patch.object(queue,'QueueStatus',side_effect=status),mock.patch.object(queue.time,'sleep',side_effect=sleep):
            self.assertEqual(self.run_queue(evaluate=evaluate,watch=True)[0],130)
        self.assertEqual(self.calls,['user1_v1.onnx','user2_v1.onix'])
        self.assertIn('종료됨',(self.root/'queue/status.txt').read_text(encoding='utf-8'))

    def test_idle_status_says_ready_but_stopped_status_does_not(self):
        self.add('user1_v1.onnx.part');self.add('.hidden.onnx')
        status=queue.QueueStatus(self.root,watch=True,poll_seconds=15)
        status.activity('ready')
        text=(self.root/'queue/status.txt').read_text(encoding='utf-8')
        self.assertIn('바로 사용 가능',text)
        self.assertIn('대기 중인 모델: 0건',text)
        self.assertIn('전송 중(.part): 1건',text)
        self.assertIn('파일명/형식 확인 필요: 1건',text)
        status.close(reason='test stop')
        text=(self.root/'queue/status.txt').read_text(encoding='utf-8')
        self.assertIn('종료됨',text);self.assertNotIn('바로 사용 가능',text)

    def test_status_excludes_completed_and_failed_jobs_from_waiting_count(self):
        for name in ['user1_v1.onnx','user2_v1.onnx','user3_v1.onnx']:self.add(name)
        snapshots,_=queue.scan_models(self.inbox)
        jobs=[{**item,'id':queue.job_key(item,{}),'status':phase}
              for item,phase in zip(snapshots,['completed','failed','pending'])]
        status=queue.QueueStatus(self.root,watch=True,poll_seconds=15)
        status.sync_jobs(snapshots,jobs,{})
        status.activity('ready')
        text=(self.root/'queue/status.txt').read_text(encoding='utf-8')
        self.assertIn('대기 중인 모델: 1건',text)
        self.assertIn('1. user3_v1.onnx',text)
        self.assertIn('누적 완료: 1건',text)
        self.assertIn('누적 실패(자동 반복 안 함): 1건',text)

    def test_unchanged_models_are_not_rehashed_at_every_idle_poll(self):
        self.add('user1_v1.onnx');cache={}
        with mock.patch.object(queue,'_sha256',wraps=queue._sha256) as digest:
            first,_=queue.scan_models(self.inbox,cache=cache)
            second,_=queue.scan_models(self.inbox,cache=cache)
            self.assertEqual(digest.call_count,1)
            self.assertEqual(first,second)
            self.add('user1_v1.onnx',b'a new model version')
            third,_=queue.scan_models(self.inbox,cache=cache)
            self.assertEqual(digest.call_count,2)
            self.assertNotEqual(first[0]['model_sha256'],third[0]['model_sha256'])

    def test_status_atomic_replace_retries_a_transient_windows_reader(self):
        status=queue.QueueStatus(self.root,watch=True,poll_seconds=15)
        replace=queue.os.replace
        attempts=[]
        def sharing_lock(source,destination):
            attempts.append(1)
            if len(attempts)==1:raise PermissionError('temporary sharing violation')
            return replace(source,destination)
        with mock.patch.object(queue.os,'name','nt'),mock.patch.object(queue.os,'replace',side_effect=sharing_lock), \
             mock.patch.object(queue.time,'sleep') as pause:
            status.refresh()
        self.assertEqual(len(attempts),2)
        pause.assert_called_once_with(.025)
        self.assertIn('RT-DETR',(self.root/'queue/status.txt').read_text(encoding='utf-8'))

    def test_disconnected_console_does_not_abort_queue_or_lose_disk_log(self):
        self.add('user1_v1.onnx');self.add('user2_v1.onnx')
        broken=mock.Mock()
        broken.write.side_effect=OSError(errno.EIO,'terminal disconnected')
        with mock.patch.object(sys,'stdout',broken):
            with queue.console_logging(self.root):
                self.assertEqual(self.run_queue()[0],0)
        self.assertEqual(self.calls,['user1_v1.onnx','user2_v1.onnx'])
        log=(self.root/'queue/service.log').read_text(encoding='utf-8')
        self.assertIn('console disconnected',log)
        self.assertIn('user2_v1.onnx',log)
        self.assertEqual(broken.write.call_count,1)

    def test_actual_filesystem_io_error_is_not_suppressed_as_console_error(self):
        disk=mock.Mock();disk.write.side_effect=OSError(errno.EIO,'disk failure')
        console=io.StringIO()
        with self.assertRaises(OSError):queue.ResilientConsole(console,disk).write('test')
        self.assertEqual(console.getvalue(),'')

    def test_termination_signal_records_name_and_restores_handler(self):
        number=queue.signal.SIGTERM;old=queue.signal.getsignal(number)
        with queue.interrupt_signals(keep_running_on_hangup=True):
            with self.assertRaises(queue.QueueInterrupted) as captured:
                queue.signal.getsignal(number)(number,None)
            self.assertEqual(captured.exception.signal_name,'SIGTERM')
            if hasattr(queue.signal,'SIGHUP'):
                self.assertEqual(queue.signal.getsignal(queue.signal.SIGHUP),queue.signal.SIG_IGN)
        self.assertEqual(queue.signal.getsignal(number),old)

    def test_service_io_error_retains_traceback_and_errno(self):
        self.add('user1_v1.onnx')
        with self.assertRaises(OSError):
            queue.run_queue(self.root,{},mock.Mock(side_effect=OSError(errno.EIO,'data disk failure')),self.evaluate)
        failure=self.state()['service']['failure']
        self.assertEqual(failure['errno'],errno.EIO)
        self.assertIn('Traceback',failure['traceback'])
        self.assertTrue((self.root/'queue/service_failure.log').is_file())

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
        for name in ['baseline.ONNX','최적화 모델.onnx']:onnx.save(model,self.inbox/name)
        self.add('user2_broken.onnx',b'not an ONNX model')
        command=[sys.executable,str(script),'--use-current-python','--once','--no-download','--no-upload',
                 '--data-dir',os.environ['COCO_DATA_DIR'],'--threads','1']
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
            self.assertEqual(metrics['image_count'],500)
            self.assertEqual(metrics['protocol_image_count'],500)
            self.assertTrue(metrics['full_evaluation'])
            self.assertEqual(metrics['evaluation_scope'],'protocol_default')
            provenance=json.loads((Path(job['result_dir'])/'inference_provenance.json').read_text())
            self.assertEqual(len(provenance['evaluated_image_ids']),500)
            self.assertGreater(metrics['resource']['rss_peak_bytes'],0)
        again=subprocess.run(command,cwd=self.root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
        self.assertEqual([j['attempts'] for j in self.state()['jobs']],[1,1,1])
        self.assertNotIn('Stage: infer',again.stdout)
        first=json.loads((Path(jobs[0]['result_dir'])/'inference_provenance.json').read_text())
        last=json.loads((Path(jobs[2]['result_dir'])/'inference_provenance.json').read_text())
        self.assertEqual(first['evaluated_image_ids'],last['evaluated_image_ids'])


if __name__=='__main__':unittest.main()
