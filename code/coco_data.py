"""Pinned COCO preparation matching the original Colab's 512/1,000 split.

This module uses only Python's standard library. Verification never edits data.
The official source ZIPs are retained so selected JPEGs can be checked by CRC.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
import urllib.request
import zipfile
import zlib

SEED = 42
CALIBRATION_COUNT = 512
EVALUATION_COUNT = 1000
PROTOCOL = {
    'model': 'official RT-DETR v1 R50-vd COCO pretrained, unchanged',
    'source_split': 'COCO 2017 val', 'seed': SEED,
    'calibration_images': CALIBRATION_COUNT, 'evaluation_images': EVALUATION_COUNT,
}
ARCHIVES = {
    'annotations_trainval2017.zip': {
        'url': 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'bytes': 252907541,
        'sha256': '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268',
    },
    'val2017.zip': {
        'url': 'https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip',
        'bytes': 815585330,
        'sha256': '4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05',
    },
}
ANNOTATION_SOURCE_SHA256 = 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'
MANIFEST_SHA256 = '98f5798b6b039770eba78a518d5d2d9f9b2630dc463dd4f5590863d4d50295ba'
SUBSET_LOCKS = {
    'calibration': {
        'files_sha256': '7a39f946ef624f90855dca0cc23707b67597a702117fe5b385a8744e6cef8706',
        'annotation_sha256': '1039e2fa8fe429b851786173e883977459998d67f5d168912836efc9e3942e60',
    },
    'evaluation': {
        'files_sha256': 'de09ee5784e4f4e8b7b34254584cd2c924dfa87ca7c1c83e648e0aa7c616b7bc',
        'annotation_sha256': '274a6265f8d4fccbe7ef08be3118c10b428389795b71d3c22d3565a341e660fd',
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_file_name(name):
    if not isinstance(name, str) or re.fullmatch(r'[0-9]{12}\.jpg', name) is None:
        raise ValueError(f'Unsafe COCO image name: {name!r}')
    return name


def _reject_symlink(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Symlinks are not accepted in the managed dataset: {path}')


def _data_root(data_dir):
    root = Path(data_dir).expanduser().absolute()
    _reject_symlink(root)
    for directory in ['downloads', 'annotations', 'val2017']:
        _reject_symlink(root / directory)
    return root


def canonical_bytes(value):
    # Deliberately no trailing newline: byte-compatible with the original Colab.
    return json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')


def _save_json(path, value):
    path = Path(path)
    _reject_symlink(path)
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Existing dataset metadata differs; refusing to overwrite: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    _reject_symlink(temporary)
    temporary.write_bytes(payload)
    temporary.replace(path)


def selected_subsets(source):
    images, categories = source['images'], source['categories']
    if len(images) != 5000 or len(categories) != 80:
        raise ValueError('Expected COCO 2017 val: 5,000 images and 80 categories')
    ordered = sorted(images, key=lambda image: _safe_file_name(image['file_name']))
    if len({image['file_name'] for image in ordered}) != 5000 or len({image['id'] for image in ordered}) != 5000:
        raise ValueError('Duplicate COCO image IDs or filenames')
    chosen = random.Random(SEED).sample(ordered, CALIBRATION_COUNT + EVALUATION_COUNT)
    selections = [('calibration', sorted(chosen[:CALIBRATION_COUNT], key=lambda image: image['file_name'])),
                  ('evaluation', sorted(chosen[CALIBRATION_COUNT:], key=lambda image: image['file_name']))]
    subsets = {}
    for name, chosen_images in selections:
        ids = {image['id'] for image in chosen_images}
        subsets[name] = {
            'info': {'source': 'COCO 2017 val', 'selection_seed': SEED,
                     'purpose': name, 'note': 'Disjoint research subset; not the full official val result'},
            'licenses': source.get('licenses', []), 'images': chosen_images,
            'annotations': [ann for ann in source['annotations'] if ann['image_id'] in ids],
            'categories': categories,
        }
    return subsets


@contextmanager
def _prepare_lock(data):
    data.mkdir(parents=True, exist_ok=True)
    lock = data / '.prepare.lock'
    _reject_symlink(lock)
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Another preparation may be active: {lock}. If its process has stopped, remove that lock and retry.') from exc
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump({'pid': os.getpid(), 'created_unix': time.time()}, output)
        yield
    finally:
        # Only remove the lock created by this call, never a directory tree.
        _reject_symlink(lock)
        lock.unlink(missing_ok=True)


def _download(data, name):
    spec = ARCHIVES[name]
    target = data / 'downloads' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink(target)
    if target.exists():
        if target.stat().st_size != spec['bytes'] or sha256(target) != spec['sha256']:
            raise ValueError(f'Existing archive failed the pinned SHA256: {target}')
        return target
    partial = target.with_suffix('.zip.part')
    _reject_symlink(partial)
    if partial.exists() and partial.stat().st_size == spec['bytes']:
        if sha256(partial) != spec['sha256']:
            raise ValueError(f'Completed partial archive failed SHA256: {partial}')
        partial.replace(target)
        return target
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > spec['bytes']:
            raise ValueError(f'Partial archive is too large: {partial}')
        headers = {'User-Agent': 'RT-DETR-RaspberryPi-Eval/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            request = urllib.request.Request(spec['url'], headers=headers)
            # HTTPS uses the platform's normal certificate/hostname verification.
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset:
                    expected = f'bytes {offset}-{spec["bytes"] - 1}/{spec["bytes"]}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not honor the requested range; partial file preserved')
                elif response.status != 200:
                    raise ValueError(f'Unexpected HTTP status {response.status}')
                with partial.open('ab' if offset else 'wb') as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            if partial.stat().st_size != spec['bytes']:
                raise OSError(f'Incomplete download: {partial.stat().st_size}/{spec["bytes"]}')
            if sha256(partial) != spec['sha256']:
                raise ValueError(f'Downloaded archive failed the pinned SHA256: {partial}')
            partial.replace(target)
            print(f'Downloaded and verified {name}', flush=True)
            return target
        except OSError as exc:
            if attempt == 4:
                raise RuntimeError(f'Download failed; rerun to resume {partial}') from exc
            print(f'Download interrupted; retry {attempt + 1}/5: {type(exc).__name__}', flush=True)
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f'Download failed: {name}')


def _crc(path):
    value = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value = zlib.crc32(block, value)
    return value & 0xffffffff


def _extract_images(archive, destination, names):
    destination.mkdir(parents=True, exist_ok=True)
    _reject_symlink(destination)
    with zipfile.ZipFile(archive) as zipped:
        for index, name in enumerate(sorted(names), 1):
            _safe_file_name(name)
            member = zipped.getinfo('val2017/' + name)
            target = destination / name
            _reject_symlink(target)
            if target.exists():
                if target.stat().st_size != member.file_size or _crc(target) != member.CRC:
                    raise ValueError(f'Existing image differs from the pinned archive: {target}')
            else:
                temporary = target.with_name(target.name + f'.{os.getpid()}.part')
                _reject_symlink(temporary)
                with zipped.open(member) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                temporary.replace(target)
            if index % 200 == 0 or index == len(names):
                print(f'COCO images prepared: {index}/{len(names)}', flush=True)


def verify_data(data_dir):
    """Read-only verification of existing data, returning paths and provenance.

    When the pinned val ZIP is retained, every selected JPEG is checked against
    its CRC. If absent, current JPEG hashes are still fingerprinted, but source
    payload verification is explicitly marked unavailable.
    """
    data = _data_root(data_dir)
    manifest_path = data / 'manifest.json'
    _reject_symlink(manifest_path)
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError('COCO manifest SHA256 differs from the original Colab split')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if any(manifest.get(k) != v for k, v in PROTOCOL.items()):
        raise ValueError('COCO selection protocol differs from the original Colab')
    if manifest['archive_sha256'] != {n: s['sha256'] for n, s in ARCHIVES.items()} or manifest['subsets'] != SUBSET_LOCKS:
        raise ValueError('COCO data lock differs from the original Colab')
    subsets = {}
    for name, expected_count in [('calibration', CALIBRATION_COUNT), ('evaluation', EVALUATION_COUNT)]:
        path = data / 'annotations' / f'{name}.json'
        _reject_symlink(path)
        if sha256(path) != SUBSET_LOCKS[name]['annotation_sha256']:
            raise ValueError(f'{name} annotation SHA256 differs from the fixed subset')
        subset = json.loads(path.read_text(encoding='utf-8'))
        images = subset['images']
        if len(images) != expected_count or len(subset['categories']) != 80:
            raise ValueError(f'Unexpected {name} subset size or category count')
        for image in images:
            _safe_file_name(image['file_name'])
        files_hash = hashlib.sha256('\n'.join(i['file_name'] for i in images).encode()).hexdigest()
        if files_hash != SUBSET_LOCKS[name]['files_sha256']:
            raise ValueError(f'{name} selected image list changed')
        subsets[name] = subset
    cal_ids = {i['id'] for i in subsets['calibration']['images']}
    eval_ids = {i['id'] for i in subsets['evaluation']['images']}
    if len(cal_ids) != CALIBRATION_COUNT or len(eval_ids) != EVALUATION_COUNT or cal_ids & eval_ids:
        raise ValueError('Duplicate or overlapping calibration/evaluation image IDs')
    verified_archives = []
    for name, spec in ARCHIVES.items():
        archive = data / 'downloads' / name
        _reject_symlink(archive)
        if archive.exists():
            if archive.stat().st_size != spec['bytes'] or sha256(archive) != spec['sha256']:
                raise ValueError(f'Existing archive failed pinned SHA256: {archive}')
            verified_archives.append(name)
    image_zip = data / 'downloads' / 'val2017.zip'
    zipped = zipfile.ZipFile(image_zip) if image_zip.exists() else None
    fingerprint = hashlib.sha256()
    try:
        images = sorted(subsets['calibration']['images'] + subsets['evaluation']['images'], key=lambda i: i['file_name'])
        for image in images:
            name = image['file_name']
            path = data / 'val2017' / name
            _reject_symlink(path)
            if not path.is_file():
                raise FileNotFoundError(f'Missing COCO image: {path}')
            if zipped:
                member = zipped.getinfo('val2017/' + name)
                if path.stat().st_size != member.file_size or _crc(path) != member.CRC:
                    raise ValueError(f'COCO image differs from pinned source ZIP: {name}')
            fingerprint.update(f'{name}\t{sha256(path)}\n'.encode())
    finally:
        if zipped:
            zipped.close()
    return {
        'annotation_path': data / 'annotations' / 'evaluation.json',
        'images_dir': data / 'val2017', 'manifest_path': manifest_path,
        'evaluation_images': subsets['evaluation']['images'],
        'calibration_images': subsets['calibration']['images'],
        'verification': {
            'manifest_sha256': MANIFEST_SHA256,
            'evaluation_annotation_sha256': SUBSET_LOCKS['evaluation']['annotation_sha256'],
            'selected_image_payloads_sha256': fingerprint.hexdigest(),
            'verified_source_archives': verified_archives,
            'selected_images_crc_verified': image_zip.exists(),
            'note': ('Selected JPEGs match the pinned source ZIP CRCs.' if image_zip.exists()
                     else 'Source image ZIP absent: JPEG hashes recorded, source CRC comparison unavailable.'),
        },
    }


def prepare_data(data_dir, download=True):
    """Prepare the exact shared data, or only verify it when download=False."""
    data = _data_root(data_dir)
    if not download:
        return verify_data(data)
    if (data / 'manifest.json').exists():
        try:
            return verify_data(data)
        except FileNotFoundError:
            # Missing images may be restored; mismatched metadata/content fails.
            pass
    with _prepare_lock(data):
        annotation_zip = _download(data, 'annotations_trainval2017.zip')
        with zipfile.ZipFile(annotation_zip) as zipped:
            source_bytes = zipped.read('annotations/instances_val2017.json')
        if hashlib.sha256(source_bytes).hexdigest() != ANNOTATION_SOURCE_SHA256:
            raise ValueError('Source COCO annotation content failed pinned SHA256')
        subsets = selected_subsets(json.loads(source_bytes))
        del source_bytes
        for name, subset in subsets.items():
            actual = hashlib.sha256(canonical_bytes(subset)).hexdigest()
            if actual != SUBSET_LOCKS[name]['annotation_sha256']:
                raise ValueError(f'{name} selection is not byte-compatible with the original Colab')
        names = {image['file_name'] for s in subsets.values() for image in s['images']}
        if len(names) != CALIBRATION_COUNT + EVALUATION_COUNT:
            raise ValueError('Calibration/evaluation image names overlap')
        image_zip = _download(data, 'val2017.zip')
        _extract_images(image_zip, data / 'val2017', names)
        for name, subset in subsets.items():
            _save_json(data / 'annotations' / f'{name}.json', subset)
        manifest = {
            **PROTOCOL,
            'archive_sha256': {name: spec['sha256'] for name, spec in ARCHIVES.items()},
            'annotation_source_sha256': ANNOTATION_SOURCE_SHA256,
            'subsets': SUBSET_LOCKS,
        }
        if hashlib.sha256(canonical_bytes(manifest)).hexdigest() != MANIFEST_SHA256:
            raise ValueError('Generated data manifest is not byte-compatible with the original Colab')
        _save_json(data / 'manifest.json', manifest)
    return verify_data(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'coco')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    result = prepare_data(args.data_dir, download=not args.verify_only)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in result.items() if k not in ['evaluation_images', 'calibration_images']}, indent=2))


if __name__ == '__main__':
    main()
