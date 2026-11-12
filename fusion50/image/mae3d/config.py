"""Flat YAML defaults, explicit CLI overrides, and startup validation."""
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def parse_config(parser, argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config')
    config_path = probe.parse_known_args(argv)[0].config
    if config_path:
        path = Path(config_path).expanduser().resolve()
        config = yaml.safe_load(path.read_text()) or {}
        if not isinstance(config, dict):
            parser.error('YAML config must be a mapping')
        actions = {a.dest:a for a in parser._actions}
        unknown = set(config) - set(actions)
        if unknown:
            parser.error(f'Unknown YAML keys: {sorted(unknown)}')
        for key, value in config.items():
            action = actions[key]
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
                if type(value) is not bool:
                    parser.error(f'{key} must be a YAML boolean')
            elif action.nargs == 3:
                if not isinstance(value, list) or len(value) != 3 or any(type(v) is not int for v in value):
                    parser.error(f'{key} must contain exactly three integers')
            elif action.type is int and type(value) is not int:
                parser.error(f'{key} must be an integer')
            elif action.type is float and value is not None and (type(value) not in (int,float)):
                parser.error(f'{key} must be numeric')
            elif action.type is str and value is not None and not isinstance(value,str):
                parser.error(f'{key} must be a string')
            if action.choices and value not in action.choices:
                parser.error(f'{key} must be one of {action.choices}')
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for key in ['data_path','output_dir','log_dir','resume']:
        value = getattr(args,key)
        if value:
            p = Path(value).expanduser()
            setattr(args,key,str((ROOT / p).resolve() if not p.is_absolute() else p.resolve()))
    for key in ['batch_size','accum_iter','epochs','embed_dim','depth','num_heads','decoder_embed_dim','decoder_depth','decoder_num_heads','in_chans','save_freq']:
        if getattr(args,key) <= 0:
            parser.error(f'{key} must be positive')
    if args.num_workers < 0 or args.max_steps < 0 or args.start_epoch < 0 or args.warmup_epochs < 0:
        parser.error('num_workers, max_steps, start_epoch and warmup_epochs must be nonnegative')
    if not 0 < args.mask_ratio < 1:
        parser.error('mask_ratio must lie strictly between 0 and 1 for training')
    if any(i <= 0 or p <= 0 or i % p for i,p in zip(args.input_size,args.patch_size)):
        parser.error('input_size must be positive and divisible by patch_size on every axis')
    for dim,heads in [(args.embed_dim,args.num_heads),(args.decoder_embed_dim,args.decoder_num_heads)]:
        if dim % 6 or dim % heads:
            parser.error('embedding dimensions must be divisible by 6 and their attention head count')
    if not Path(args.data_path).is_dir():
        parser.error(f'Data directory does not exist: {args.data_path}; set data_path in YAML or --data_path')
    if args.resume and not Path(args.resume).is_file():
        parser.error(f'Resume checkpoint does not exist: {args.resume}')
    return args
