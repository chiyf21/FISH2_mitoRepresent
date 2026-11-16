'''Export native Concat1536 and Project384 on the fixed Fish2 Reviewed81 cohort.'''
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .checkpoint import free_gpu
from .export_nohead_reviewed81 import reviewed_inputs, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-gpu', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    free_gpu(args.physical_gpu)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'fusion50/refine'))
    from train import core, load_models

    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if (checkpoint['stage'], checkpoint['epoch'], checkpoint['step']) != ('refine', 5, 2875):
        raise ValueError('Expected the matched refine E5 checkpoint')
    config = checkpoint['config']
    balance = config.get('pair_balance', False)
    points_config = dict(config, pointclouds=str(Path(config['point_run']) / 'data/pointclouds.npy'))
    target, groups, points, mapping = reviewed_inputs(points_config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'input_rows.json').write_text(json.dumps(mapping) + '\n')
    device = torch.device('cuda:0')
    torch.set_num_threads(1)
    torch.manual_seed(20261111)
    torch.cuda.manual_seed_all(20261111)
    torch.backends.cuda.matmul.allow_tf32 = False
    point, grouping, image = load_models(config, device)
    model = core.System(point, image).to(device)
    if balance:
        model.balance = torch.nn.ModuleDict({
            str(layer): core.PairBalance(point.trans_dim) for layer in core.LAYERS
        }).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.phase = 'refine'
    model.eval().requires_grad_(False)
    buffers = {key: x.clone() for key, x in model.balance.state_dict().items()} if balance else {}
    vectors = [np.empty((len(target), dim), np.float32) for dim in (1536, 384)]
    seen = np.zeros(len(target), dtype=bool)

    @torch.inference_mode()
    def infer(rows, cache):
        tokens = core.prepare(rows, cache, point, grouping, image, device, 0, False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p, i, fused, concat = model(tokens, views=1, loss=False)
        if not balance:
            assert torch.equal(concat[12], torch.cat((p[12], i[12]), dim=-1))
        arrays = [x[12][:, 0].float().cpu().numpy() for x in (concat, fused)]
        assert all(x.shape == (len(rows), dim) and np.isfinite(x).all()
                   for x, dim in zip(arrays, (1536, 384)))
        return arrays

    smoke_ids = {}
    for source, rows in groups.items():
        probe = rows[:8]
        smoke_ids[source] = [row['record_id'] for row in probe]
        assert all(np.array_equal(x, y) for x, y in
                   zip(infer(probe, points[source]), infer(probe, points[source])))
        print('deterministic smoke passed', source, flush=True)
        if args.smoke:
            continue
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            indices = [row['embedding_row'] for row in batch]
            assert not seen[indices].any()
            for dest, src in zip(vectors, infer(batch, points[source])):
                dest[indices] = src
            seen[indices] = True
            if start % 512 == 0 or start + 8 >= len(rows):
                print('exported', int(seen.sum()), 'of', len(target), flush=True)

    if balance:
        assert all(torch.equal(x, buffers[k]) for k, x in model.balance.state_dict().items())
    if not args.smoke:
        assert seen.all()
        for name, x in zip(('concat_12', 'fused_12'), vectors):
            np.save(args.output / (name + '.npy'), x)
        target.to_csv(args.output / 'instance_map.csv', index=False)
    receipt = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                   layer=12, dimensions=[1536, 384], exported=int(seen.sum()),
                   pair_balance=balance, smoke_ids=smoke_ids, frozen_balance=True)
    (args.output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
