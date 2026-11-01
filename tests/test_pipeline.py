"""End-to-end runs of scripts/slam.py on a synthetic sequence (CPU, stand-in rasterizer).

Checks that the full system runs and writes everything the table scripts read: tracking, mapping every
N frames, densification, keyframes, pruning, evaluation, mechanism metrics, timing and the final map,
for VCS-SLAM, the SGS-SLAM baseline, an ablation, a Tab. IX variant and corrupted priors.
"""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, STUBS_DIR
from synthetic_scene import smoke_overrides

RUNS = {
    "synth0_0": ("configs/replica/slam.py", []),
    "synth0_0_sgs_slam": ("configs/replica/sgs_slam.py", []),
    "synth0_0_no_vcsu": ("configs/replica/slam.py", ["semantics.vcsu=false"]),
    "synth0_0_base": ("configs/replica/slam.py", ["semantics.vcsu=false", "semantics.scea=false", "semantics.cauw=false"]),
    "synth0_0_entropy": ("configs/replica/slam.py", ["semantics.cauw_variant=entropy"]),
    "synth0_0_sgs_slam_uniform0.2": ("configs/replica/sgs_slam.py",
                                     ["semantics.prior_corruption.kind=uniform", "semantics.prior_corruption.level=0.2"]),
}


def slam(config, output_root, data_root, yaml_path, tag=None, extra=()):
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "slam.py"), config, "--scene", "synth0"]
    if tag:
        cmd += ["--tag", tag]
    cmd += ["--set", *smoke_overrides(data_root, output_root, yaml_path), *extra]
    env = dict(os.environ, PYTHONPATH=STUBS_DIR, MPLBACKEND="Agg")
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)


@pytest.fixture(scope="module")
def runs(synthetic_sequence, tmp_path_factory):
    data_root, yaml_path = synthetic_sequence
    output_root = str(tmp_path_factory.mktemp("experiments"))
    for name, (config, extra) in RUNS.items():
        base = "synth0_0_sgs_slam" if "sgs_slam" in config else "synth0_0"
        tag = name[len(base) + 1:] or None
        result = slam(config, output_root, data_root, yaml_path, tag, extra)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        assert "Failed" not in result.stdout, result.stdout[-3000:]
    return os.path.join(output_root, "Replica")


@pytest.mark.parametrize("name", list(RUNS))
def test_run_outputs(runs, name):
    run_dir = os.path.join(runs, name)
    metrics = json.load(open(os.path.join(run_dir, "eval", "metrics.json")))
    for key in ("ate_rmse_cm", "psnr", "ssim", "lpips", "depth_l1_cm", "miou_percent"):
        assert metrics[key] is not None and math.isfinite(metrics[key]), key
    assert metrics["num_eval_frames"] == 7
    mechanism = json.load(open(os.path.join(run_dir, "eval", "mechanism.json")))
    assert 0.0 <= mechanism["boundary_f1"] <= 1.0 and 0.0 <= mechanism["error_auroc"] <= 1.0
    timing = json.load(open(os.path.join(run_dir, "eval", "timing.json")))
    assert timing["num_frames"] == 12 and timing["slam_fps"] > 0
    assert timing["num_mapping_steps"] == (12 if "sgs_slam" in name else 2)    # every frame vs. every N = 8 frames
    resolved = json.load(open(os.path.join(run_dir, "config.json")))
    assert resolved["run_name"] == name
    params = np.load(os.path.join(run_dir, "params.npz"))
    num_gaussians = params["means3D"].shape[0]
    assert params["semantic_ids"].shape == (num_gaussians,) and params["semantic_colors"].shape == (num_gaussians, 3)
    if "sgs_slam" in name:
        assert "semantic_logits" not in params.files
    else:
        assert params["semantic_logits"].shape == (num_gaussians, len(params["label_space_ids"]))
        assert params["label_space_ids"].tolist() == [0, 3, 7, 12, 20, 25]
    labels = os.listdir(os.path.join(run_dir, "eval", "rendered_label"))
    assert len(labels) == 7 and os.path.exists(os.path.join(run_dir, "params.ply"))


def test_skip_existing(runs, synthetic_sequence):
    data_root, yaml_path = synthetic_sequence
    result = slam("configs/replica/slam.py", os.path.dirname(runs), data_root, yaml_path,
                  extra=["--skip-existing"])
    assert result.returncode == 0 and "complete, skipping" in result.stdout


def tool(args, runs, synthetic_sequence):
    env = dict(os.environ, PYTHONPATH=STUBS_DIR, MPLBACKEND="Agg", VCS_OUTPUT_ROOT=os.path.dirname(runs),
               VCS_DATA_ROOT=synthetic_sequence[0])
    result = subprocess.run([sys.executable, *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result.stdout


def row(output, prefix):
    cells = next(line for line in output.splitlines() if line.startswith(prefix)).strip("|").split("|")
    return [c.strip() for c in cells]


def test_tables(runs, synthetic_sequence):
    out = tool(["tools/make_tables.py", "table7", "--scenes", "synth0"], runs, synthetic_sequence)
    for module in ("VCSU", "SCEA", "CAUW"):
        assert "-" not in row(out, f"| {module}")[2:4]                   # SGS-SLAM and VCS-SLAM columns filled
    out = tool(["tools/make_tables.py", "table6", "--scenes", "synth0"], runs, synthetic_sequence)
    assert "-" not in row(out, "| without VCSU module")[1:] and "-" not in row(out, "| With All modules")[1:]
    assert row(out, "| without SCEA module")[1:] == ["-"] * 4               # not run in this test
    out = tool(["tools/make_tables.py", "table1", "--scenes", "synth0"], runs, synthetic_sequence)
    metrics = json.load(open(os.path.join(runs, "synth0_0", "eval", "metrics.json")))
    assert row(out, "| VCS-SLAM (Ours)")[1] == f"{metrics['ate_rmse_cm']:.2f}"
    tool(["tools/make_tables.py", "all", "--scenes", "synth0"], runs, synthetic_sequence)


def test_figures(runs, synthetic_sequence, tmp_path):
    tool(["tools/make_figures.py", "fig4", "--scenes", "synth0", "--frame", "1", "--crop", "5", "5", "25", "20",
          "--baseline", "sgs_slam", "--out", str(tmp_path)], runs, synthetic_sequence)
    tool(["tools/make_figures.py", "fig4", "--scenes", "synth0", "--frame", "1", "--out", str(tmp_path / "base")],
         runs, synthetic_sequence)
    tool(["tools/make_figures.py", "fig3", "--scenes", "synth0", "--frame", "1", "--out", str(tmp_path)],
         runs, synthetic_sequence)
    assert all(os.path.exists(p) for p in (tmp_path / "fig3.png", tmp_path / "fig4.png", tmp_path / "base" / "fig4.png"))


def test_experiment_commands(runs, synthetic_sequence):
    lines = tool(["tools/experiments.py", "commands", "table8", "--scenes", "synth0"], runs, synthetic_sequence)
    lines = lines.strip().splitlines()
    assert len(lines) == 2 * (1 + 12)                                       # 2 methods x (clean + 4 kinds x 3 levels)
    assert all(line.startswith("python scripts/slam.py configs/replica/") and line.endswith("--skip-existing")
               for line in lines)
    assert "--tag uniform0.2 --set semantics.prior_corruption.kind=uniform semantics.prior_corruption.level=0.2" \
           in "\n".join(lines)
