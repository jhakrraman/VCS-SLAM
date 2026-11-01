"""The experiments of the paper: which SLAM runs each table needs, and how to launch them.

    python tools/experiments.py list                    # experiments and their number of runs
    python tools/experiments.py run table6              # run whatever is missing, then print Tab. VI
    python tools/experiments.py commands table6         # one scripts/slam.py command per line (for job arrays)

Every run writes to $VCS_OUTPUT_ROOT/<Replica|ScanNet>/<scene>_<seed>[_<tag>]; finished runs are skipped,
so an interrupted experiment can simply be launched again. ``--scenes`` and ``--seed`` override the defaults.
"""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from vcs_slam.config import load_config  # noqa: E402

REPLICA = "configs/replica/slam.py"
REPLICA_SGS = "configs/replica/sgs_slam.py"
SCANNET = "configs/scannet/slam.py"

REPLICA_SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
SEMANTIC_SCENES = ["room0", "room1", "room2", "office0"]          # Tab. V protocol
SCANNET_SCENES = ["scene0000_00", "scene0059_00", "scene0106_00", "scene0169_00", "scene0181_00", "scene0207_00"]

ABLATIONS = {                                                    # Tab. VI
    "": [],
    "no_vcsu": ["semantics.vcsu=false"],
    "no_scea": ["semantics.scea=false"],
    "no_cauw": ["semantics.cauw=false"],
}
CORRUPTIONS = {                                                  # Tab. VIII: mild / moderate / severe
    "boundary": [1, 3, 5],             # k px
    "confusion": [0.05, 0.10, 0.20],   # p
    "missed": [0.10, 0.25, 0.50],      # q
    "uniform": [0.05, 0.10, 0.20],     # r (control)
}
VARIANTS = {                                                     # Tab. IX
    "hardgate": ["semantics.vcsu_variant=hard"],
    "tv": ["semantics.scea_variant=tv"],
    "entropy": ["semantics.cauw_variant=entropy"],
}
STRUCTURAL = ("boundary", 3)                                     # Tab. IX "Struct." column
BASE_MODEL = ("base", ["semantics.vcsu=false", "semantics.scea=false", "semantics.cauw=false"])  # Fig. 4 (b)
SENSITIVITY = {                                                  # Sec. IV-D, on Room-0
    "sigma_vis": [0.025, 0.05, 0.1],
    "gamma": [50.0, 100.0, 200.0],
    "tau_d": [0.5, 0.75, 0.9],
}
PAPER_DEFAULTS = {"sigma_vis": 0.05, "gamma": 100.0, "tau_d": 0.75}


def corruption_tag(kind, level):
    return f"{kind}{level:g}"


def corruption_overrides(kind, level):
    return [f"semantics.prior_corruption.kind={kind}", f"semantics.prior_corruption.level={level}"]


@dataclass
class Run:
    config: str
    scene: str
    tag: str = ""
    overrides: list = field(default_factory=list)

    def command(self, seed, python=sys.executable):
        cmd = [python, "scripts/slam.py", self.config, "--scene", self.scene, "--seed", str(seed)]
        if self.tag:
            cmd += ["--tag", self.tag]
        if self.overrides:
            cmd += ["--set", *self.overrides]
        return cmd + ["--skip-existing"]

    def run_dir(self, seed):
        config = load_config(os.path.join(REPO_ROOT, self.config), self.scene, seed, self.tag or None)
        return os.path.join(config["workdir"], config["run_name"])


def full_runs(config, scenes):
    return [Run(config, s) for s in scenes]


def experiment_runs(name, scenes=None):
    """The runs behind each table / figure of the paper."""
    if name in ("table1", "table3", "table4", "fig3"):
        return full_runs(REPLICA, scenes or REPLICA_SCENES)
    if name == "table5":
        return full_runs(REPLICA, scenes or SEMANTIC_SCENES)
    if name == "table2":
        return full_runs(SCANNET, scenes or SCANNET_SCENES)
    if name == "table6":
        return [Run(REPLICA, s, tag, sets) for tag, sets in ABLATIONS.items() for s in scenes or REPLICA_SCENES]
    if name == "table7":
        return full_runs(REPLICA, scenes or REPLICA_SCENES) + full_runs(REPLICA_SGS, scenes or REPLICA_SCENES)
    if name == "fig4":
        scenes = scenes or ["room0"]
        return full_runs(REPLICA, scenes) + [Run(REPLICA, s, *BASE_MODEL) for s in scenes]
    if name == "table8":
        runs = []
        for config in (REPLICA_SGS, REPLICA):
            runs += full_runs(config, scenes or SEMANTIC_SCENES)
            runs += [Run(config, s, corruption_tag(kind, level), corruption_overrides(kind, level))
                     for kind, levels in CORRUPTIONS.items() for level in levels for s in scenes or SEMANTIC_SCENES]
        return runs
    if name == "table9":
        kind, level = STRUCTURAL
        runs = []
        for variant, sets in [("", [])] + list(VARIANTS.items()):
            for s in scenes or SEMANTIC_SCENES:
                runs.append(Run(REPLICA, s, variant, sets))
                struct_tag = "_".join(t for t in (variant, corruption_tag(kind, level)) if t)
                runs.append(Run(REPLICA, s, struct_tag, sets + corruption_overrides(kind, level)))
        return runs
    if name == "sensitivity":
        runs = []
        for s in scenes or ["room0"]:
            runs.append(Run(REPLICA, s))
            runs += [Run(REPLICA, s, f"{param}{value:g}", [f"semantics.{param}={value}"])
                     for param, values in SENSITIVITY.items() for value in values if value != PAPER_DEFAULTS[param]]
        return runs
    raise ValueError(f"Unknown experiment {name!r}; see `python tools/experiments.py list`.")


EXPERIMENTS = {
    "table1": "Tab. I    ATE RMSE on Replica",
    "table2": "Tab. II   ATE RMSE on ScanNet",
    "table3": "Tab. III  depth L1 and runtime on Replica",
    "table4": "Tab. IV   rendering quality on Replica",
    "table5": "Tab. V    semantic mIoU on Replica",
    "table6": "Tab. VI   module ablation on Replica",
    "table7": "Tab. VII  mechanism analysis (VCS-SLAM vs. SGS-SLAM)",
    "table8": "Tab. VIII robustness to corrupted 2D priors",
    "table9": "Tab. IX   naive reweighting variants",
    "sensitivity": "Sec. IV-D sensitivity to sigma_vis, gamma, tau_d on Room-0",
    "fig3": "Fig. 3    RGB renderings (Room-0, Room-1, Office-0)",
    "fig4": "Fig. 4    qualitative ablation (base model without the modules vs. VCS-SLAM)",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["list", "run", "commands"])
    parser.add_argument("experiment", nargs="?", choices=list(EXPERIMENTS))
    parser.add_argument("--scenes", nargs="+", help="Scenes to run (default: those of the paper)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.action == "list":
        for name, description in EXPERIMENTS.items():
            print(f"{name:12s} {len(experiment_runs(name)):4d} runs   {description}")
        return
    if args.experiment is None:
        parser.error("name an experiment, e.g. table6")
    runs = experiment_runs(args.experiment, args.scenes)
    if args.action == "commands":
        for run in runs:
            print(shlex.join(run.command(args.seed, python="python")))
        return

    failed = []
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {shlex.join(run.command(args.seed))}", flush=True)
        if subprocess.run(run.command(args.seed), cwd=REPO_ROOT).returncode != 0:
            failed.append(run)
    if failed:
        print(f"{len(failed)} run(s) failed:", *(shlex.join(r.command(args.seed)) for r in failed), sep="\n  ")
    target = {"fig3": None, "fig4": None}.get(args.experiment, args.experiment)
    if target:
        subprocess.run([sys.executable, "tools/make_tables.py", target, "--seed", str(args.seed)]
                       + (["--scenes", *args.scenes] if args.scenes else []), cwd=REPO_ROOT)
    else:
        print(f"Renderings are in the eval/ folder of each run; compose the panel with "
              f"`python tools/make_figures.py {args.experiment}`.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
