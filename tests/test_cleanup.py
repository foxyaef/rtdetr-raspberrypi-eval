"""Deletion is limited to byte-matching successful uploads."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import github_upload as uploader
import model_queue


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cleanup_');self.root=Path(self.temp.name)
        self.model=self.root/'queue/inbox/user1_v1.onnx';self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b'uploaded model')
        self.results_root=self.root/'results';self.output=self.results_root/'user1_v1_run';self.output.mkdir(parents=True)
        sha=uploader._sha256(self.model)
        (self.output/'provenance.json').write_text(json.dumps({'model_name':self.model.name,'model_sha256':sha}),encoding='utf8')
        (self.output/'evaluation_state.json').write_text('{"status":"complete"}',encoding='utf8')
        (self.output/'metrics.json').write_text('{}')
        self.archive=Path(str(self.output)+'.zip');self.archive.write_bytes(b'uploaded ZIP')
        hashes={p.name:uploader._sha256(p) for p in self.output.iterdir()}
        hashes['results.zip']=uploader._sha256(self.archive)
        self.receipt={'status':'uploaded','commit_sha':'a'*40,'model_sha256':sha,'branch':'user1',
                      'run_id':self.output.name,'uploaded_files_sha256':hashes}
        self.other=self.root/'queue/inbox/user2_v1.onnx';self.other.write_bytes(b'keep model')
        (self.results_root/'other_run').mkdir()

    def tearDown(self):self.temp.cleanup()

    def cleanup(self):
        return uploader.cleanup_uploaded_run(self.root,self.results_root,self.model,self.output,self.archive,self.receipt)

    def test_success_removes_only_selected_files_and_retains_durable_receipt(self):
        record=self.cleanup()
        self.assertEqual(record['cleanup_status'],'complete')
        self.assertFalse(self.model.exists());self.assertFalse(self.output.exists());self.assertFalse(self.archive.exists())
        self.assertTrue(self.other.exists());self.assertTrue((self.results_root/'other_run').is_dir())
        self.assertTrue((self.root/'queue/uploaded/user1_v1_run.json').is_file())
        self.assertEqual(model_queue.load_state(self.root/'queue')['jobs'][0]['model_name'],self.model.name)

    def test_failed_upload_or_changed_model_or_changed_zip_never_deletes(self):
        for change in ('failure','model','zip','result'):
            with self.subTest(change=change):
                original_model=self.model.read_bytes();original_zip=self.archive.read_bytes()
                original_metric=(self.output/'metrics.json').read_bytes()
                if change=='failure':self.receipt['status']='failed'
                elif change=='model':self.model.write_bytes(b'replacement')
                elif change=='zip':self.archive.write_bytes(b'replacement')
                else:(self.output/'metrics.json').write_bytes(b'replacement')
                with self.assertRaises(uploader.UploadError):self.cleanup()
                self.assertTrue(self.model.exists() and self.output.exists() and self.archive.exists())
                self.receipt['status']='uploaded';self.model.write_bytes(original_model);self.archive.write_bytes(original_zip)
                (self.output/'metrics.json').write_bytes(original_metric)

    def test_path_outside_selected_results_root_is_refused(self):
        with self.assertRaises(uploader.UploadError):
            uploader.cleanup_uploaded_run(self.root,self.root/'different',self.model,self.output,self.archive,self.receipt)
        self.assertTrue(self.model.exists() and self.archive.exists())

    def test_partial_cleanup_preserves_model_and_records_failure(self):
        with mock.patch.object(uploader.shutil,'rmtree',side_effect=OSError('permission denied')):
            record=self.cleanup()
        self.assertEqual(record['cleanup_status'],'partial');self.assertTrue(self.model.exists())
        self.assertIn('permission denied',record['cleanup_error'])

    def test_restart_recovers_completion_and_status_keeps_model_name(self):
        job={'id':'old-job','model_name':self.model.name,'model_sha256':self.receipt['model_sha256'],
             'result_dir':str(self.output),'status':'running','attempts':1}
        model_queue.save_state(self.root/'queue',{'format':1,'jobs':[job]})
        self.cleanup()
        state=model_queue.load_state(self.root/'queue')
        self.assertEqual(state['jobs'][0]['status'],'completed')
        status=model_queue.QueueStatus(self.root,watch=True,poll_seconds=15)
        status.sync_jobs([],state['jobs'],{});status.activity('ready')
        text=(self.root/'queue/status.txt').read_text(encoding='utf8')
        self.assertIn('완료된 모델:\n  user1_v1.onnx',text)
        self.assertIn('대기 중인 모델: 1건',text)  # Unrelated user2 model is still waiting.

    def test_once_does_not_reject_its_own_deleted_input_or_pick_other_models(self):
        def evaluate(name,digest,prepared,progress):
            record=self.cleanup()
            return {'returncode':0,'upload_status':'uploaded','cleanup_status':record['cleanup_status'],
                    'upload_receipt':record['upload_receipt'],'result_dir':str(self.output),'archive_path':str(self.archive)}
        result=model_queue.run_queue(self.root,{},lambda:object(),evaluate,names=[self.model.name])
        self.assertEqual(result,0)
        state=model_queue.load_state(self.root/'queue')
        self.assertEqual(state['rejected'],[])
        self.assertTrue(self.other.is_file())


if __name__=='__main__':unittest.main()
