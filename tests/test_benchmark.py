"""Benchmark semantics and actual CPU ONNX/COCO subprocess integration."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'code'
spec = importlib.util.spec_from_file_location('pi_benchmark_under_test', CODE/'benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_category_mapping_and_xyxy_conversion_keep_low_scores_for_AP(self):
        import numpy as np
        outputs = [np.array([[0, 1]], dtype=np.int64),
                   np.array([[[1, 2, 5, 8], [2, 3, 4, 6]]], dtype=np.float32),
                   np.array([[.9, .01]], dtype=np.float32)]
        records = benchmark.decode(outputs, {'id': 42}, [1, 3])
        self.assertEqual([r['category_id'] for r in records], [1, 3])
        self.assertEqual(records[0]['bbox'], [1., 2., 4., 6.])
        self.assertEqual(records[1]['bbox'], [2., 3., 2., 3.])
        self.assertLess(records[1]['score'], .5)
        self.assertEqual([r['image_id'] for r in records], [42, 42])

    def test_decode_rejects_invalid_shapes_and_values(self):
        import numpy as np
        labels = np.array([[0]], dtype=np.int64)
        boxes = np.array([[[1, 2, 5, 8]]], dtype=np.float32)
        scores = np.array([[.9]], dtype=np.float32)
        invalid = [
            [labels.reshape(1), boxes, scores],
            [labels.astype(float), boxes, scores],
            [np.array([[2]]), boxes, scores],
            [labels, boxes.reshape(1, 4), scores],
            [labels, np.full_like(boxes, np.nan), scores],
            [labels, boxes, np.array([[1.01]])],
            [labels, boxes, np.array([[-.01]])],
        ]
        for outputs in invalid:
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                benchmark.decode(outputs, {'id': 42}, [1, 3])

    def test_decode_excludes_zero_area_and_zero_score(self):
        import numpy as np
        outputs = [np.array([[0, 0, 0]], dtype=np.int64),
                   np.array([[[1, 1, 1, 5], [0, 0, 4, 4], [5, 1, 2, 4]]], dtype=np.float32),
                   np.array([[.9, 0, .9]], dtype=np.float32)]
        self.assertEqual(benchmark.decode(outputs, {'id': 42}, [1]), [])

    @staticmethod
    def session(images_shape=None, images_type='tensor(float)', size_type='tensor(int64)', output_names=None):
        inputs = [SimpleNamespace(name='images', shape=images_shape or [1, 3, 640, 640], type=images_type),
                  SimpleNamespace(name='orig_target_sizes', shape=[1, 2], type=size_type)]
        outputs = [SimpleNamespace(name=n) for n in (output_names or ['labels', 'boxes', 'scores'])]
        return SimpleNamespace(get_inputs=lambda: inputs, get_outputs=lambda: outputs)

    def test_deploy_interface_rejects_wrong_shape_dtype_or_outputs(self):
        for session in [self.session(images_shape=[1, 3, 320, 320]),
                        self.session(images_shape=[2, 3, 640, 640]),
                        self.session(images_type='tensor(int8)'),
                        self.session(size_type='tensor(float)'),
                        self.session(output_names=['labels', 'boxes'])]:
            with self.subTest(session=session), self.assertRaises(ValueError):
                benchmark.check_interface(session)

    def test_deploy_interface_accepts_symbolic_dims_and_expected_dtypes(self):
        import numpy as np
        result = benchmark.check_interface(self.session(images_shape=['batch', 3, 'height', 'width'],
                                                        images_type='tensor(float16)', size_type='tensor(int32)'))
        self.assertEqual(result, (np.float16, np.int32))

    def test_percentiles_match_linear_interpolation(self):
        self.assertEqual(benchmark.distribution([]), {'mean': None, 'p50': None, 'p95': None})
        self.assertEqual(benchmark.distribution([7]), {'mean': 7, 'p50': 7, 'p95': 7})
        value = benchmark.distribution([4, 1, 3, 2])
        self.assertEqual(value['mean'], 2.5)
        self.assertEqual(value['p50'], 2.5)
        self.assertAlmostEqual(value['p95'], 3.85)

    def test_CPU_peak_keeps_per_core_units_and_machine_normalization(self):
        fake_process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100),
                                       cpu_times=lambda: SimpleNamespace(user=10., system=2.))
        with patch('psutil.Process', return_value=fake_process), patch('psutil.cpu_count', return_value=4), \
             patch('psutil.virtual_memory', return_value=SimpleNamespace(total=1000, available=400)):
            sampler = benchmark.ResourceSampler('unused.csv')
            sampler.started = 10.
            sampler.cpu_start = SimpleNamespace(user=8., system=1.)
            sampler.cpu_peak = 350.
            with patch.object(benchmark.time, 'perf_counter', return_value=12.):
                measured = sampler.finish()
        self.assertEqual(measured['cpu_process_peak_percent_one_core_100'], 350.)
        self.assertEqual(measured['cpu_process_peak_percent_machine_normalized'], 87.5)
        self.assertEqual(measured['cpu_process_mean_percent_one_core_100'], 150.)
        self.assertEqual(measured['system_memory_used_peak_bytes'], 600)

    def test_image_preprocessing_retains_RGB_range_and_width_height_order(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            (data/'val2017').mkdir()
            name = '000000000001.jpg'
            Image.new('RGB', (24, 16), (128, 64, 32)).save(data/'val2017'/name, format='PNG')
            item = {'id': 1, 'file_name': name, 'width': 24, 'height': 16}
            images, sizes = benchmark.load_image(item, data, np.float32, np.int64)
        self.assertEqual(images.shape, (1, 3, 640, 640))
        self.assertEqual(sizes.tolist(), [[24, 16]])
        self.assertAlmostEqual(float(images[0, 0, 0, 0]), 128/255, places=6)
        self.assertAlmostEqual(float(images[0, 1, 0, 0]), 64/255, places=6)
        self.assertAlmostEqual(float(images[0, 2, 0, 0]), 32/255, places=6)
        self.assertTrue(images.flags.c_contiguous)

    def test_actual_ONNX_workers_score_once_and_report_resources(self):
        if not (CODE/'coco_metrics.py').is_file():
            self.skipTest('COCO scoring helper is not available yet')
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            data, output = temp/'data', temp/'results'
            (data/'val2017').mkdir(parents=True)
            (data/'annotations').mkdir()
            output.mkdir()
            category_ids = [1, 3] + list(range(5, 83))
            images = []
            annotations = []
            for index, image_id in enumerate([101, 202], 1):
                name = f'{image_id:012d}.jpg'
                Image.new('RGB', (16, 12), (70, 90, 110)).save(data/'val2017'/name)
                images.append({'id': image_id, 'file_name': name, 'width': 16, 'height': 12})
                annotations.append({'id': index, 'image_id': image_id, 'category_id': 1,
                                    'bbox': [2, 2, 6, 6], 'area': 36, 'iscrowd': 0})
            ann = {'info': {'description': 'Small integration fixture'}, 'licenses': [],
                   'images': images, 'annotations': annotations,
                   'categories': [{'id': cid, 'name': f'class_{cid}'} for cid in category_ids]}
            (data/'annotations/evaluation.json').write_text(json.dumps(ann), encoding='utf-8')
            values = {'labels': np.array([[0, 1]], dtype=np.int64),
                      'boxes': np.array([[[2, 2, 8, 8], [1, 1, 5, 5]]], dtype=np.float32),
                      'scores': np.array([[.9, .1]], dtype=np.float32)}
            nodes = [helper.make_node('Constant', [], [name], value=numpy_helper.from_array(value))
                     for name, value in values.items()]
            inputs = [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640]),
                      helper.make_tensor_value_info('orig_target_sizes', TensorProto.INT64, [1, 2])]
            outputs = [helper.make_tensor_value_info('labels', TensorProto.INT64, [1, 2]),
                       helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 2, 4]),
                       helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 2])]
            model = helper.make_model(helper.make_graph(nodes, 'constant_detector_fixture', inputs, outputs),
                                      opset_imports=[helper.make_opsetid('', 17)])
            model.ir_version = 10
            model_path = temp/'user1_fixture.onnx'
            onnx.save(model, model_path)
            config = {'model_path': str(model_path), 'data_dir': str(data), 'result_dir': str(output),
                      'limit': 2, 'threads': 1, 'repeats': 3}
            config_path = temp/'config.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            for worker in ['audit', 'infer', 'score']:
                completed = subprocess.run([sys.executable, str(CODE/'test.py'), '--worker', worker,
                                            '--config', str(config_path)], cwd=temp,
                                           capture_output=True, text=True, timeout=120)
                self.assertEqual(completed.returncode, 0,
                                 worker+'\n'+completed.stdout+'\n'+completed.stderr)
            audit = json.loads((output/'quantization_audit.json').read_text(encoding='utf-8'))
            resource = json.loads((output/'inference_metrics.json').read_text(encoding='utf-8'))
            provenance = json.loads((output/'inference_provenance.json').read_text(encoding='utf-8'))
            accuracy = json.loads((output/'accuracy.json').read_text(encoding='utf-8'))
            predictions = json.loads((output/'predictions.json').read_text(encoding='utf-8'))
            self.assertTrue(audit['self_contained'])
            self.assertEqual(provenance['providers'], ['CPUExecutionProvider'])
            self.assertEqual(provenance['evaluated_image_ids'], [101, 202])
            self.assertEqual(resource['inference_runs_timed'], 6)
            self.assertEqual(resource['inference_images_timed'], 2)
            self.assertEqual(resource['latency_repeats'], 3)
            self.assertEqual(provenance['prediction_count'], 4)
            self.assertEqual(len(predictions), 4)  # Repeated inference must not triple AP predictions.
            self.assertEqual({p['category_id'] for p in predictions}, {1, 3})
            self.assertAlmostEqual(accuracy['mAP50_95'], 1., places=6)
            self.assertGreater(resource['rss_peak_bytes'], 0)
            self.assertGreaterEqual(resource['cpu_process_peak_percent_one_core_100'], 0)
            self.assertIsNone(resource['power']['power_mean_watts'])
            self.assertIsNone(resource['power']['power_peak_watts'])
            self.assertIsNone(resource['power']['energy_joules'])
            for timing in resource['latency'].values():
                self.assertGreaterEqual(timing['mean'], 0)
                self.assertGreaterEqual(timing['p95'], timing['p50'])
            with (output/'latency.csv').open(encoding='utf-8') as stream:
                timing_rows = list(csv.DictReader(stream))
            self.assertEqual(len(timing_rows), 2)
            self.assertTrue((output/'resource_samples.csv').is_file())
            with (output/'class_metrics.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 80)


if __name__ == '__main__':
    unittest.main()
