"""SGS-SLAM baseline on Replica (Tabs. VII and VIII), run inside this code base.

    python scripts/slam.py configs/replica/sgs_slam.py --scene room0

Identical to SGS-SLAM's original Replica configuration: 3-channel semantic colours with an L1 + SSIM
loss, no reliability modules, mapping on every frame and SGS-SLAM's semantic loss weights.
Everything else is shared with configs/replica/slam.py. Runs are tagged "sgs_slam".
"""

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "vcs_slam_replica_config", os.path.join(os.path.dirname(os.path.abspath(__file__)), "slam.py"))
_vcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vcs)

SCENES = _vcs.SCENES
SEMANTIC_SCENES = _vcs.SEMANTIC_SCENES


def get_config(scene_name="room0", seed=0, tag=""):
    config = _vcs.get_config(scene_name, seed, "sgs_slam" + (f"_{tag}" if tag else ""))
    config["map_every"] = 1
    config["semantics"].update(representation="color", vcsu=False, scea=False, cauw=False)
    config["tracking"]["loss_weights"]["seg"] = 0.05
    config["mapping"]["loss_weights"]["seg"] = 0.1
    for stage, lr in (("tracking", 0.0), ("mapping", 0.0025)):
        lrs = config[stage]["lrs"]
        del lrs["semantic_logits"]
        lrs["semantic_colors"] = lr
    return config


config = get_config()
