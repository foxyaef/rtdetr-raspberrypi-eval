"""Safety checks and an actual offline Git/LFS branch upload integration."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import github_upload as uploader


def git(args, cwd=None):
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    r = subprocess.run(["git", *args], cwd=cwd, env=env, text=True, capture_output=True, check=False)
    if r.returncode:
        raise AssertionError(r.stderr)
    return r.stdout.strip()


def make_model(path):
    import onnx
    from onnx import helper, TensorProto
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "test",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path.write_bytes(model.SerializeToString())
    return model


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uploader_validation_")
        self.root = Path(self.temp.name)
        self.model = self.root / "user1_v1.onnx"
        make_model(self.model)
        self.run = self.root / "run_001"
        self.run.mkdir()
        (self.run / "metrics.json").write_text('{"mAP50_95": 0.5}', encoding="utf-8")
        (self.run / "provenance.json").write_text(json.dumps({"model_name":self.model.name,
            "model_sha256":hashlib.sha256(self.model.read_bytes()).hexdigest()}),encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_identity_and_legacy_extension(self):
        self.assertEqual(uploader.infer_user(self.model), "user1")
        self.assertEqual(uploader.infer_user("user4_decoder_v2.onix"), "user4")
        for name in ["user5_v1.onnx", "fp32.onnx", "user1_../secret.onnx", "user1_v1.exe", "user1_..v1.onnx"]:
            with self.subTest(name=name), self.assertRaises(uploader.UploadError):
                uploader.infer_user(name)

    def test_wrong_branch_and_credential_url_rejected_before_git(self):
        with mock.patch.object(uploader, "_run") as run:
            with self.assertRaises(uploader.UploadError):
                uploader.upload_run(self.model, self.run, branch="user2")
            with self.assertRaises(uploader.UploadError):
                uploader.upload_run(self.model, self.run, repo="https://token@github.com/owner/repo")
            run.assert_not_called()

    def test_credential_file_rejected(self):
        (self.run / ".env").write_text("GITHUB_TOKEN=do-not-upload", encoding="utf-8")
        with self.assertRaises(uploader.UploadError):
            uploader._validate_inputs(self.model, self.run, "owner/repo", None)

    def test_unknown_result_type_rejected(self):
        (self.run / "private.key").write_text("do-not-upload", encoding="utf-8")
        with self.assertRaises(uploader.UploadError):
            uploader._validate_inputs(self.model, self.run, "owner/repo", None)

    def test_symlink_escape_rejected(self):
        target = self.root / "unrelated.txt"
        target.write_text("outside results", encoding="utf-8")
        try:
            (self.run / "outside.txt").symlink_to(target)
        except OSError:
            self.skipTest("This Windows account cannot create symbolic links")
        with self.assertRaises(uploader.UploadError):
            uploader._validate_inputs(self.model, self.run, "owner/repo", None)

    def test_external_data_rejected_without_reading_external_file(self):
        import onnx
        model = make_model(self.model)
        tensor = model.graph.initializer.add()
        tensor.name = "external"
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.dims.append(1)
        tensor.data_location = onnx.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key = "location"
        entry.value = "../unrelated-secret.bin"
        self.model.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(uploader.UploadError, "External-weight"):
            uploader._validate_inputs(self.model, self.run, "owner/repo", None)

    def test_secret_error_text_is_redacted(self):
        token = "ghp_never_print_this_secret123456"
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": token}):
            text = uploader._sanitize(f"fatal https://name:{token}@github.com/owner/repo {token}")
        self.assertNotIn(token, text)
        self.assertIn("[redacted]", text)
        self.assertNotIn("base64secret", uploader._sanitize("Authorization: Basic base64secret"))

    def test_inherited_http_tracing_is_removed(self):
        with mock.patch.dict(os.environ,{"GIT_CURL_VERBOSE":"1","GIT_TRACE_CURL":"1"}):
            environment=uploader._environment()
        self.assertNotIn("GIT_CURL_VERBOSE",environment)
        self.assertNotIn("GIT_TRACE_CURL",environment)
        self.assertEqual(environment["GIT_LFS_SKIP_SMUDGE"],"1")

    def test_wrong_model_provenance_rejected_before_git(self):
        (self.run / "provenance.json").write_text('{"model_name":"other.onnx","model_sha256":"wrong"}',encoding="utf-8")
        with mock.patch.object(uploader,"_run") as run:
            with self.assertRaisesRegex(uploader.UploadError,"does not match"):
                uploader.upload_run(self.model,self.run,"owner/repo")
            run.assert_not_called()


@unittest.skipUnless(shutil.which("git"), "git is required")
class LocalGitLFSIntegration(unittest.TestCase):
    def setUp(self):
        try:
            git(["lfs", "version"])
        except AssertionError:
            self.skipTest("git-lfs is required for the offline integration")
        self.temp = tempfile.TemporaryDirectory(prefix="uploader_git_")
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote.git"
        git(["init", "--bare", "--initial-branch=main", str(self.remote)])
        seed = self.root / "seed"
        git(["init", "--initial-branch=main", str(seed)])
        git(["config", "user.name", "Test"], seed)
        git(["config", "user.email", "test@example.invalid"], seed)
        (seed / "README.md").write_text("Benchmark code stays on main.\n", encoding="utf-8")
        (seed / ".gitignore").write_text("models/*\nresults/\n", encoding="utf-8")
        git(["add", "README.md", ".gitignore"], seed)
        git(["commit", "-m", "Initial source"], seed)
        git(["remote", "add", "origin", str(self.remote)], seed)
        git(["push", "origin", "main"], seed)
        self.main_sha = git(["rev-parse", "main"], self.remote)
        self.model = self.root / "user1_v1.onnx"
        make_model(self.model)
        self.run = self.make_run("run_001")
        self.remote_patch = mock.patch.object(uploader, "_remote_url", return_value=str(self.remote))
        self.remote_patch.start()

    def tearDown(self):
        if hasattr(self, "remote_patch"):
            self.remote_patch.stop()
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def make_run(self, name):
        path = self.root / name
        path.mkdir()
        (path / "metrics.json").write_text(json.dumps({"run": name, "mAP50_95": 0.5}), encoding="utf-8")
        (path / "provenance.json").write_text(json.dumps({"model_name":self.model.name,
            "model_sha256":hashlib.sha256(self.model.read_bytes()).hexdigest()}),encoding="utf-8")
        (path / "logs").mkdir()
        (path / "logs" / "evaluation.log").write_text("evaluation complete\n", encoding="utf-8")
        return path

    def test_real_lfs_upload_retains_history_and_does_not_change_main(self):
        cwd = Path.cwd()
        first = uploader.upload_run(self.model, self.run, "owner/repo")
        self.assertEqual(Path.cwd(), cwd)
        self.assertEqual(first["branch"], "user1")
        self.assertEqual(git(["rev-parse", "main"], self.remote), self.main_sha)
        pointer = git(["show", "user1:models/user1_v1.onnx"], self.remote)
        digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.assertIn("oid sha256:" + digest, pointer)
        self.assertIn("version https://git-lfs.github.com/spec/v1", pointer)
        # A real LFS payload must have reached the bare remote, not only a pointer.
        objects = list((self.remote / "lfs" / "objects").rglob(digest))
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].read_bytes(), self.model.read_bytes())
        second_run = self.make_run("run_002")
        second = uploader.upload_run(self.model, second_run, "owner/repo")
        self.assertNotEqual(first["commit_sha"], second["commit_sha"])
        self.assertIn("run_001", git(["show", "user1:results/user1/run_001/metrics.json"], self.remote))
        self.assertIn("run_002", git(["show", "user1:results/user1/run_002/metrics.json"], self.remote))
        self.assertEqual(git(["rev-parse", "main"], self.remote), self.main_sha)
        self.assertTrue(self.model.is_file() and self.run.is_dir())

    def test_duplicate_run_is_not_overwritten(self):
        uploader.upload_run(self.model, self.run, "owner/repo")
        before = git(["rev-parse", "user1"], self.remote)
        identical = uploader.upload_run(self.model, self.run, "owner/repo")
        self.assertTrue(identical["already_present"])
        self.assertEqual(git(["rev-parse", "user1"], self.remote), before)
        (self.run / "upload_status.json").write_text('{"status":"uploaded"}',encoding="utf-8")
        receipt_retry = uploader.upload_run(self.model, self.run, "owner/repo")
        self.assertTrue(receipt_retry["already_present"])
        self.assertEqual(git(["rev-parse", "user1"], self.remote), before)
        (self.run / "metrics.json").write_text('{"changed":true}',encoding="utf-8")
        with self.assertRaisesRegex(uploader.UploadError, "already exists"):
            uploader.upload_run(self.model, self.run, "owner/repo")
        self.assertEqual(git(["rev-parse", "user1"], self.remote), before)
        self.assertTrue((self.run / "metrics.json").is_file())

    def test_zip_upload_and_retry_match_local_archive(self):
        archive=Path(str(self.run)+'.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(self.run.rglob('*')):
                if path.is_file():bundle.write(path,path.relative_to(self.run).as_posix())
        receipt=uploader.upload_run(self.model,self.run,'owner/repo',archive_path=archive)
        clone=self.root/'zip_check'
        git(['clone','--branch','user1',str(self.remote),str(clone)])
        remote_archive=clone/'results/user1/run_001/results.zip'
        self.assertEqual(remote_archive.read_bytes(),archive.read_bytes())
        (self.run/'upload_status.json').write_text('{"status":"uploaded"}',encoding='utf-8')
        retried=uploader.upload_run(self.model,self.run,'owner/repo',archive_path=archive)
        self.assertTrue(retried['already_present'])
        self.assertEqual(retried['commit_sha'],receipt['commit_sha'])
        (self.run/'metrics.json').write_text('{"changed":true}',encoding='utf-8')
        with self.assertRaisesRegex(uploader.UploadError,'ZIP contents differ'):
            uploader.upload_run(self.model,self.run,'owner/repo',archive_path=archive)

    def test_concurrent_fast_forward_is_rebased_without_force(self):
        uploader.upload_run(self.model, self.run, "owner/repo")
        other = self.root / "other"
        git(["clone", "--no-checkout", str(self.remote), str(other)])
        git(["checkout", "-b", "user1", "origin/user1"], other)
        git(["config", "user.name", "Peer"], other)
        git(["config", "user.email", "peer@example.invalid"], other)
        next_run = self.make_run("run_003")
        original = uploader._run
        pushed_once = False

        def race(args, cwd=None, *, check=True):
            nonlocal pushed_once
            if args[0] == "push" and not pushed_once:
                pushed_once = True
                (other / "peer.txt").write_text("Concurrent benchmark metadata\n", encoding="utf-8")
                git(["add", "peer.txt"], other)
                git(["commit", "-m", "Peer result"], other)
                git(["push", "origin", "user1"], other)
            if args[0] == "push":
                self.assertNotIn("--force", args)
            return original(args, cwd, check=check)

        with mock.patch.object(uploader, "_run", side_effect=race):
            result = uploader.upload_run(self.model, next_run, "owner/repo")
        self.assertEqual(result["status"], "uploaded")
        self.assertIn("Concurrent", git(["show", "user1:peer.txt"], self.remote))
        self.assertIn("run_003", git(["show", "user1:results/user1/run_003/metrics.json"], self.remote))
        self.assertEqual(git(["rev-parse", "main"], self.remote), self.main_sha)


if __name__ == "__main__":
    unittest.main()
