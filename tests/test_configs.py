"""Experiment configs: paper hyper-parameters, the SGS-SLAM baseline, and command-line overrides."""

import os

import pytest

from conftest import REPO_ROOT
from vcs_slam.config import apply_overrides, load_config, parse_value

REPLICA = os.path.join(REPO_ROOT, "configs", "replica", "slam.py")
REPLICA_SGS = os.path.join(REPO_ROOT, "configs", "replica", "sgs_slam.py")
SCANNET = os.path.join(REPO_ROOT, "configs", "scannet", "slam.py")


@pytest.mark.parametrize("path", [REPLICA, SCANNET])
def test_vcs_slam_configs_use_the_paper_settings(path):
    """Sec. IV-A4 of the paper."""
    cfg = load_config(path)
    sem = cfg["semantics"]
    assert sem["representation"] == "logits"
    assert sem["vcsu"] and sem["scea"] and sem["cauw"]
    assert sem["sigma_vis"] == 0.05 and sem["beta"] == 0.7 and sem["gamma"] == 100.0
    assert sem["lambda_edge"] == 0.02 and sem["tau_d"] == 0.75
    assert sem["prior_corruption"]["kind"] is None
    assert cfg["map_every"] == 8
    assert cfg["tracking"]["loss_weights"] == {"im": 0.5, "depth": 1.0, "seg": 0.0}
    assert cfg["mapping"]["loss_weights"] == {"im": 0.5, "depth": 1.0, "seg": 0.01}
    assert cfg["mechanism"] == {"inconsistent_depth": 0.05, "boundary_tolerance": 3}
    assert cfg["seed"] == 0 and not cfg["use_wandb"] and not cfg["load_checkpoint"]
    assert cfg["checkpoint_time_idx"] == 0


def test_scene_lists():
    from vcs_slam.config import load_module
    assert load_module(REPLICA).SCENES == ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    assert load_module(REPLICA).SEMANTIC_SCENES == ["room0", "room1", "room2", "office0"]
    assert load_module(SCANNET).SCENES == ["scene0000_00", "scene0059_00", "scene0106_00", "scene0169_00",
                                          "scene0181_00", "scene0207_00"]


def test_sgs_slam_baseline_is_sgs_slams_replica_config():
    """Values of SGS-SLAM's original configs/replica/slam.py; everything else is shared with VCS-SLAM."""
    sgs, vcs = load_config(REPLICA_SGS), load_config(REPLICA)
    assert sgs["semantics"]["representation"] == "color"
    assert not (sgs["semantics"]["vcsu"] or sgs["semantics"]["scea"] or sgs["semantics"]["cauw"])
    assert sgs["map_every"] == 1 and sgs["keyframe_every"] == 5 and sgs["mapping_window_size"] == 24
    assert sgs["tracking"]["num_iters"] == 40 and sgs["mapping"]["num_iters"] == 60
    assert sgs["tracking"]["loss_weights"] == {"im": 0.5, "depth": 1.0, "seg": 0.05}
    assert sgs["mapping"]["loss_weights"] == {"im": 0.5, "depth": 1.0, "seg": 0.1}
    assert sgs["tracking"]["lrs"] == dict(means3D=0.0, rgb_colors=0.0, unnorm_rotations=0.0, logit_opacities=0.0,
                                          log_scales=0.0, cam_unnorm_rots=0.0004, cam_trans=0.002,
                                          semantic_colors=0.0)
    assert sgs["mapping"]["lrs"] == dict(means3D=0.0001, rgb_colors=0.0025, unnorm_rotations=0.001,
                                         logit_opacities=0.05, log_scales=0.001, cam_unnorm_rots=0.0,
                                         cam_trans=0.0, semantic_colors=0.0025)
    assert sgs["run_name"] == "room0_0_sgs_slam"
    for key in ("data", "mechanism", "keyframe_every", "eval_every", "scene_radius_depth_ratio", "viz"):
        assert sgs[key] == vcs[key]
    for stage in ("tracking", "mapping"):
        strip = {k: v for k, v in vcs[stage].items() if k not in ("loss_weights", "lrs")}
        assert {k: v for k, v in sgs[stage].items() if k not in ("loss_weights", "lrs")} == strip


def test_scene_seed_and_tag_set_every_derived_field():
    cfg = load_config(REPLICA, scene="office3", seed=2, tag="no_vcsu")
    assert cfg["run_name"] == "office3_2_no_vcsu" and cfg["seed"] == 2
    assert cfg["data"]["sequence"] == "office3" and cfg["viz"]["scene_name"] == "office3"
    assert cfg["wandb"]["name"] == "office3_2_no_vcsu"
    assert load_config(REPLICA_SGS, scene="room1", tag="boundary3")["run_name"] == "room1_0_sgs_slam_boundary3"


def test_overrides():
    cfg = load_config(REPLICA, overrides=["semantics.vcsu=false", "semantics.sigma_vis=0.025", "semantics.eps=1e-8",
                                          "semantics.cauw_variant=entropy", "semantics.prior_corruption.kind=boundary",
                                          "semantics.prior_corruption.level=3", "map_every=1"])
    sem = cfg["semantics"]
    assert sem["vcsu"] is False and sem["sigma_vis"] == 0.025 and sem["eps"] == 1e-8
    assert sem["cauw_variant"] == "entropy" and cfg["map_every"] == 1
    assert sem["prior_corruption"] == {"kind": "boundary", "level": 3, "seed": 0, "small_object_frac": 0.01}
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["semantics.sigma=0.1"])          # typo
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["semantics.vcsu"])                # no value


@pytest.mark.parametrize("raw,value", [("true", True), ("False", False), ("none", None), ("3", 3), ("0.75", 0.75),
                                       ("1e-8", 1e-8), ("[1, 2]", [1, 2]), ("room0", "room0"), ("hard", "hard")])
def test_parse_value(raw, value):
    assert parse_value(raw) == value
