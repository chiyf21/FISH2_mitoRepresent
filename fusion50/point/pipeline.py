"""Prepare a run directory for distributed surface Point-MAE training."""
import argparse
import json
from pathlib import Path

from data import prepare, split_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Prepare data and config before starting train_ddp.py')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parent
    cfg['point_mae_repo'] = str((root / cfg['point_mae_repo']).resolve())
    if cfg['normalization'] != 'unit_sphere':
        raise ValueError('Surface Point-MAE requires unit-sphere normalization')
    spacing = cfg['spacing_xyz_nm']
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError('Expected positive XYZ spacing')
    source = Path(cfg['input']['path']).resolve()
    cfg['input']['path'] = str(source)
    source_config = json.loads((source / 'dataset_manifest.json').read_text())
    if 'surface' not in source_config['point_sampling'].lower():
        raise ValueError('Expected a surface point cache')
    out = Path(cfg['output']).resolve()
    cfg['output'] = str(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = prepare(cfg, out / 'data')
    write_json(out / 'split.json', split_rows(rows, cfg))
    write_json(out / 'config.resolved.json', cfg)
    print(f'Prepared {len(rows)} instances in {out}')


if __name__ == '__main__':
    main()
