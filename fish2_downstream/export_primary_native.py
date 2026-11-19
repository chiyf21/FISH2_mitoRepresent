"""Export matched native Fusion50 checkpoints on a frozen Fish2 panel."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'exp/fusion50/native768'))
from train import core, load_models

READOUTS = {'point': 768, 'image': 768, 'concat': 1536, 'projection': 384}
SEED = 20261105


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def prepare(panel, reference, checkpoint, out, workers):
    manifest = read_json(panel / 'dataset_manifest.json')
    for name, digest in manifest['sha256'].items():
        if sha(panel / name) != digest:
            raise ValueError(f'Dataset file changed: {name}')
    frame = pd.read_csv(panel / 'instances.csv', dtype={'body_id': str})
    assert len(frame) == manifest['instance_count'] and frame.record_id.is_unique
    if 'neurons_per_cell_type' in manifest:
        expected = manifest['neurons_per_cell_type']
        assert frame.groupby('cell_type').body_id.nunique().eq(expected).all()
    assert frame.body_id.nunique() == manifest['neuron_count']
    assert frame.voxel_size_xyz_nm.eq('8,8,30').all() and frame.array_axis_order.eq('ZYX').all()
    cohort = read_json(reference / 'cohort.json')
    cached = {r['key'][2]: r for r in cohort}
    body_split = {r['body_id']: r['split'] for r in cohort}
    assert all(body_split[r['body_id']] == r['split'] for r in cohort)
    frame['refine_split'] = frame.body_id.map(body_split).fillna('external')
    frame['record_in_refine_cohort'] = frame.record_id.isin(cached)
    frame['embedding_row'] = np.arange(len(frame))
    frame.to_csv(out / 'instance_map.csv', index=False)
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cache = np.load(Path(ck['config']['point_run']) / 'data/pointclouds.npy', mmap_mode='r')
    masks = out / 'filled_masks'
    masks.mkdir()
    points = np.lib.format.open_memmap(out / 'pointclouds.npy', mode='w+', dtype='float32',
                                      shape=(len(frame), 2, 2048, 3))
    neighbors3d = ndimage.generate_binary_structure(3, 1)

    def prepare_row(r):
        if r.record_id in cached:
            old = cached[r.record_id]
            assert old['body_id'] == r.body_id
            return dict(old, point_row=r.embedding_row), cache[old['point_row']]
        raw = np.load(r.mask_path, mmap_mode='r', allow_pickle=False)
        em = np.load(r.em_path, mmap_mode='r', allow_pickle=False)
        if raw.ndim != 3 or raw.shape != em.shape:
            raise ValueError(f'Shape mismatch: {r.record_id}')
        mask = raw > 0
        changed = False
        for plane in mask:
            holes = ndimage.binary_fill_holes(plane) & ~plane
            if not holes.any():
                continue
            labels, _ = ndimage.label(holes)
            sizes = np.bincount(labels.ravel())
            small = np.flatnonzero((sizes > 0) & (sizes < 200))
            small = small[small > 0]
            if len(small):
                plane |= np.isin(labels, small)
                changed = True
        mask_path = r.mask_path
        if changed:
            mask_path = str(masks / f'{r.embedding_row:06d}.npy')
            np.save(mask_path, mask.astype('uint8'), allow_pickle=False)
        xyz = np.argwhere(mask & ~ndimage.binary_erosion(mask, structure=neighbors3d))[:, ::-1]
        if len(xyz) < 2:
            raise ValueError(f'Empty surface: {r.record_id}')
        rng = np.random.default_rng(SEED + r.embedding_row * 2)
        vectors = np.empty((2, 2048, 3), dtype=np.float32)
        radius = 0.
        for view in range(2):
            sample = xyz[rng.choice(len(xyz), 2048, replace=len(xyz)<2048)].astype(np.float32)
            sample -= sample.mean(axis=0, keepdims=True)
            radius = max(float(np.linalg.norm(sample, axis=1).max()), 1e-8)
            vectors[view] = sample / radius
        row = dict(key=[r.source_run_id, r.body_id, r.record_id], body_id=r.body_id,
                   cohort_index=r.embedding_row+len(cohort), point_row=r.embedding_row,
                   em_path=r.em_path, mask_path=mask_path, normalization_radius_voxels=radius,
                   spacing_xyz_nm=[8,8,30], split=r.refine_split)
        return row, vectors

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(frame), 128):
            for row, xyz in pool.map(prepare_row, frame.iloc[start:start+128].itertuples(index=False)):
                assert row['point_row'] == len(rows)
                points[len(rows)] = xyz
                rows.append(row)
            if start % 512 == 0:
                print('Prepared', len(rows), '/', len(frame), flush=True)
    points.flush()
    dump(out / 'inputs.json', rows)
    print('Prepared panel:', len(rows), flush=True)


def export(checkpoint, out, batch_size):
    rows = read_json(out / 'inputs.json')
    points = np.load(out / 'pointclouds.npy', mmap_mode='r')
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if ck['stage'] != 'refine':
        raise ValueError('Expected a refine checkpoint')
    config = ck['config']
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    torch.manual_seed(20261111)
    torch.cuda.manual_seed_all(20261111)
    torch.backends.cuda.matmul.allow_tf32 = False
    point, grouping, image = load_models(config, device)
    model = core.System(point, image).to(device)
    if config.get('pair_balance', False):
        model.balance = torch.nn.ModuleDict({str(layer):core.PairBalance(point.trans_dim)
                                            for layer in core.LAYERS}).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval().requires_grad_(False)
    model.phase = 'refine'
    target = out / checkpoint.parent.parent.name
    target.mkdir()
    vectors = {name:np.lib.format.open_memmap(target/f'{name}.partial.npy', mode='w+',
               dtype=np.float32, shape=(len(rows),dim)) for name,dim in READOUTS.items()}
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch = rows[start:start+batch_size]
            inputs = core.prepare(batch, points, point, grouping, image, device, 0, False)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p, i, fused, concat = model(inputs, views=1, loss=False)
            if model.balance is None:
                assert torch.equal(concat[12],torch.cat((p[12],i[12]),dim=-1))
            for name, features in zip(READOUTS,(p,i,concat,fused)):
                x = features[12][:,0].float().cpu().numpy()
                if not np.isfinite(x).all():
                    raise ValueError(f'Nonfinite {name} at {start}')
                vectors[name][start:start+len(batch)] = x
            if start % 512 == 0:
                print('Export',ck['epoch'],bool(model.balance),start+len(batch),'/',len(rows),flush=True)
    for name, matrix in vectors.items():
        matrix.flush()
        (target/f'{name}.partial.npy').replace(target/f'{name}.npy')
    dump(target/'complete.json',dict(checkpoint=str(checkpoint), sha256=sha(checkpoint),
                                   epoch=ck['epoch'], balance=bool(model.balance),
                                   instances=len(rows),dimensions=READOUTS))
    print('Export complete', target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    a = parser.parse_args()
    if a.prepare:
        prepare(a.panel,a.reference,a.checkpoints[0],a.output,a.workers)
    for ck in a.checkpoints:
        export(ck,a.output,a.batch_size)


if __name__ == '__main__':
    main()
