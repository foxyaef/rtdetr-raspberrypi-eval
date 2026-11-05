"""Dataset invariants and interrupted-download safety; no network needed."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SOURCE = Path(__file__).resolve().parents[1] / 'code' / 'coco_data.py'
spec = importlib.util.spec_from_file_location('coco_data_under_test', SOURCE)
coco = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coco)


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class CocoDataTests(unittest.TestCase):
    @staticmethod
    def source():
        return {
            'images': [{'id': i, 'file_name': f'{i:012d}.jpg', 'width': 1, 'height': 1} for i in range(5000)],
            'annotations': [{'id': i, 'image_id': i, 'category_id': 1} for i in range(5000)],
            'categories': [{'id': i + 1, 'name': str(i)} for i in range(80)],
        }

    def test_selection_is_order_independent_and_disjoint(self):
        original = self.source()
        first = coco.selected_subsets(original)
        original['images'].reverse()
        second = coco.selected_subsets(original)
        for name, expected in [('calibration', 512), ('evaluation', 1000)]:
            self.assertEqual(first[name]['images'], second[name]['images'])
            self.assertEqual(len(first[name]['images']), expected)
            ids = {i['id'] for i in first[name]['images']}
            self.assertEqual(ids, {a['image_id'] for a in first[name]['annotations']})
        self.assertFalse({i['id'] for i in first['calibration']['images']} & {i['id'] for i in first['evaluation']['images']})

    def test_duplicate_image_ids_are_rejected(self):
        source = self.source()
        source['images'][0]['id'] = source['images'][1]['id']
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            coco.selected_subsets(source)

    def test_archive_paths_cannot_escape_destination(self):
        for name in ['../x.jpg', '..\\x.jpg', '/x.jpg', 'C:\\x.jpg', 'val2017/x.jpg', 'x.jpg']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                coco._safe_file_name(name)

    def test_offline_verification_does_not_create_missing_data(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'missing'
            with self.assertRaises(FileNotFoundError):
                coco.prepare_data(target, download=False)
            self.assertFalse(target.exists())

    def test_existing_metadata_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'manifest.json'
            target.write_text('{"unexpected": true}', encoding='utf-8')
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                coco._save_json(target, {'expected': True})
            self.assertEqual(target.read_bytes(), before)

    def test_resume_download_checks_byte_range_and_hash(self):
        content = b'official-source-content'
        lock = {'toy.zip': {'url': 'https://example.invalid/toy.zip', 'bytes': len(content),
                            'sha256': hashlib.sha256(content).hexdigest()}}
        with tempfile.TemporaryDirectory() as folder, patch.object(coco, 'ARCHIVES', lock):
            data = Path(folder)
            (data/'downloads').mkdir()
            part = data/'downloads/toy.zip.part'
            part.write_bytes(content[:5])
            response = Response(content[5:], status=206,
                                headers={'Content-Range': f'bytes 5-{len(content)-1}/{len(content)}'})
            with patch.object(coco.urllib.request, 'urlopen', return_value=response) as opened:
                target = coco._download(data, 'toy.zip')
            self.assertEqual(target.read_bytes(), content)
            self.assertFalse(part.exists())
            self.assertEqual(opened.call_args.args[0].get_header('Range'), 'bytes=5-')

    def test_resume_refuses_server_ignoring_range_without_destroying_part(self):
        content = b'official-source-content'
        lock = {'toy.zip': {'url': 'https://example.invalid/toy.zip', 'bytes': len(content),
                            'sha256': hashlib.sha256(content).hexdigest()}}
        with tempfile.TemporaryDirectory() as folder, patch.object(coco, 'ARCHIVES', lock):
            data = Path(folder)
            (data/'downloads').mkdir()
            part = data/'downloads/toy.zip.part'
            part.write_bytes(content[:5])
            with patch.object(coco.urllib.request, 'urlopen', return_value=Response(content)), self.assertRaisesRegex(ValueError, 'range'):
                coco._download(data, 'toy.zip')
            self.assertEqual(part.read_bytes(), content[:5])

    def test_download_refuses_wrong_hash_and_preserves_file(self):
        lock = {'toy.zip': {'url': 'https://example.invalid/toy.zip', 'bytes': 3,
                            'sha256': hashlib.sha256(b'yes').hexdigest()}}
        with tempfile.TemporaryDirectory() as folder, patch.object(coco, 'ARCHIVES', lock):
            data = Path(folder)
            with patch.object(coco.urllib.request, 'urlopen', return_value=Response(b'bad')), self.assertRaisesRegex(ValueError, 'SHA256'):
                coco._download(data, 'toy.zip')
            self.assertEqual((data/'downloads/toy.zip.part').read_bytes(), b'bad')
            self.assertFalse((data/'downloads/toy.zip').exists())

    def test_crc_detects_modified_existing_image(self):
        name = '000000000001.jpg'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root/'images.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('val2017/'+name, b'original')
            images = root/'images'
            coco._extract_images(archive, images, {name})
            (images/name).write_bytes(b'modified')  # Same length, different payload.
            with self.assertRaisesRegex(ValueError, 'differs'):
                coco._extract_images(archive, images, {name})

    def test_prepare_lock_excludes_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)/'data'
            with coco._prepare_lock(data):
                with self.assertRaisesRegex(RuntimeError, 'active'):
                    with coco._prepare_lock(data):
                        self.fail('A second writer entered the lock')
                self.assertTrue((data/'.prepare.lock').exists())
            self.assertFalse((data/'.prepare.lock').exists())

    def test_symlink_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            real = root/'real.json'
            real.write_text('{}', encoding='utf-8')
            link = root/'link.json'
            try:
                link.symlink_to(real)
            except (OSError, NotImplementedError):
                self.skipTest('Symlink creation is unavailable on this host')
            with self.assertRaisesRegex(ValueError, 'Symlink'):
                coco._save_json(link, {})

    @unittest.skipUnless(os.environ.get('COCO_DATA_DIR'), 'Set COCO_DATA_DIR for offline pinned-dataset integration')
    def test_existing_real_dataset_matches_original_colab(self):
        result = coco.verify_data(Path(os.environ['COCO_DATA_DIR']))
        self.assertEqual(len(result['evaluation_images']), 1000)
        self.assertEqual(len(result['calibration_images']), 512)
        self.assertEqual(result['verification']['evaluation_annotation_sha256'],
                         '274a6265f8d4fccbe7ef08be3118c10b428389795b71d3c22d3565a341e660fd')
        self.assertEqual(result['verification']['manifest_sha256'],
                         '98f5798b6b039770eba78a518d5d2d9f9b2630dc463dd4f5590863d4d50295ba')

    @unittest.skipUnless(os.environ.get('COCO_DATA_DIR'), 'Set COCO_DATA_DIR for source-archive integration')
    def test_generation_is_byte_compatible_with_original_colab(self):
        data = Path(os.environ['COCO_DATA_DIR'])
        archive = data/'downloads/annotations_trainval2017.zip'
        if not archive.exists():
            self.skipTest('Original source annotation ZIP is not retained')
        with zipfile.ZipFile(archive) as z:
            source = json.loads(z.read('annotations/instances_val2017.json'))
        subsets = coco.selected_subsets(source)
        for name, subset in subsets.items():
            payload = coco.canonical_bytes(subset)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), coco.SUBSET_LOCKS[name]['annotation_sha256'])
            self.assertEqual(payload, (data/'annotations'/f'{name}.json').read_bytes())
        manifest = {**coco.PROTOCOL,
                    'archive_sha256': {n: s['sha256'] for n, s in coco.ARCHIVES.items()},
                    'annotation_source_sha256': coco.ANNOTATION_SOURCE_SHA256,
                    'subsets': coco.SUBSET_LOCKS}
        self.assertEqual(hashlib.sha256(coco.canonical_bytes(manifest)).hexdigest(), coco.MANIFEST_SHA256)
        self.assertEqual(coco.canonical_bytes(manifest), (data/'manifest.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
