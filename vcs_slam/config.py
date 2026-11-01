"""Experiment configs: load a config file and apply command-line overrides.

A config file defines ``get_config(scene_name=..., seed=..., tag=...)`` returning a nested dict (see
``configs/replica/slam.py``); a module-level ``config`` dict is also accepted. Any entry can then be
overridden with dotted keys, e.g. ``--set semantics.vcsu=false mapping.num_iters=30``.
"""

import ast
import copy
import importlib.util
import os


def load_module(path):
    """Import a Python file by path."""
    name = "cfg_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_value(raw):
    """Parse an override value: booleans, None, numbers and Python literals; anything else is a string."""
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def apply_overrides(config, overrides):
    """Apply ``key.sub=value`` overrides in place. Unknown keys raise, so typos do not pass silently."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override {item!r} is not of the form key=value.")
        *parents, leaf = key.strip().split(".")
        node = config
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"Unknown config key {key!r}.")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"Unknown config key {key!r}.")
        node[leaf] = parse_value(raw)
    return config


def load_config(path, scene=None, seed=None, tag=None, overrides=()):
    module = load_module(path)
    kwargs = {k: v for k, v in (("scene_name", scene), ("seed", seed), ("tag", tag)) if v is not None}
    if hasattr(module, "get_config"):
        config = module.get_config(**kwargs)
    elif kwargs:
        raise ValueError(f"{path} has no get_config(); --scene/--seed/--tag are not supported for it.")
    else:
        config = copy.deepcopy(module.config)
    return apply_overrides(config, overrides)


def add_config_args(parser):
    """Arguments shared by every script that runs or inspects an experiment."""
    parser.add_argument("config", help="Experiment config, e.g. configs/replica/slam.py")
    parser.add_argument("--scene", help="Scene / sequence name (default: the config's default scene)")
    parser.add_argument("--seed", type=int, help="Random seed (default: the config's seed)")
    parser.add_argument("--tag", help="Suffix appended to the run name, e.g. no_vcsu")
    parser.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[],
                        metavar="KEY=VALUE", help="Override config entries, e.g. --set semantics.vcsu=false")
    return parser


def config_from_args(args):
    return load_config(args.config, args.scene, args.seed, args.tag, args.overrides)
