"""Print the tables of the paper from finished runs (eval/metrics.json, mechanism.json, timing.json).

    python tools/make_tables.py table1          # any of: table1 ... table9, sensitivity, all

Missing runs are shown as "-" and left out of averages. Run them with `python tools/experiments.py run <table>`.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import (CORRUPTIONS, PAPER_DEFAULTS, REPLICA, REPLICA_SCENES, REPLICA_SGS, SCANNET,  # noqa: E402
                         SCANNET_SCENES, SEMANTIC_SCENES, SENSITIVITY, STRUCTURAL, VARIANTS, Run, corruption_tag)

SHORT = {"room0": "R0", "room1": "R1", "room2": "R2", "office0": "Of0", "office1": "Of1", "office2": "Of2",
         "office3": "Of3", "office4": "Of4"}


def load(run, seed):
    """Metrics of a finished run as one flat dict, or None."""
    eval_dir = os.path.join(run.run_dir(seed), "eval")
    path = os.path.join(eval_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    result = json.load(open(path))
    for extra in ("mechanism", "timing"):
        extra_path = os.path.join(eval_dir, f"{extra}.json")
        if os.path.exists(extra_path):
            result.update(json.load(open(extra_path)))
    return result


def value(run, key, seed):
    result = load(run, seed)
    if result is None or result.get(key) is None:
        return None
    return result[key]


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None


def fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def table(title, header, rows, note=None):
    print(f"\n{title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    if note:
        print(f"\n{note}")


def per_scene_table(title, config, scenes, key, digits, seed, scale=1.0):
    values = [value(Run(config, s), key, seed) for s in scenes]
    values = [None if v is None else v * scale for v in values]
    header = ["Method", "Avg."] + [SHORT.get(s, s.replace("scene", "").replace("_00", "")) for s in scenes]
    table(title, header, [["VCS-SLAM (Ours)", fmt(mean(values), digits)] + [fmt(v, digits) for v in values]])


def table1(seed, scenes):
    per_scene_table("Tab. I - ATE RMSE [cm] on Replica", REPLICA, scenes or REPLICA_SCENES, "ate_rmse_cm", 2, seed)


def table2(seed, scenes):
    per_scene_table("Tab. II - ATE RMSE [cm] on ScanNet", SCANNET, scenes or SCANNET_SCENES, "ate_rmse_cm", 2, seed)


def table3(seed, scenes):
    scenes = scenes or REPLICA_SCENES
    keys = [("depth_l1_cm", 3), ("tracking_fps", 2), ("mapping_fps", 2), ("slam_fps", 2)]
    row = ["VCS-SLAM (Ours)"] + [fmt(mean([value(Run(REPLICA, s), k, seed) for s in scenes]), d) for k, d in keys]
    table("Tab. III - mapping and runtime on Replica (averaged over scenes)",
          ["Method", "Depth L1 [cm]", "Track. FPS", "Map. FPS", "SLAM FPS"], [row],
          "FPS as in SGS-SLAM: 1 / tracking time per frame, 1 / time per mapping step, "
          "frames / total tracking + mapping time (see eval/timing.json).")


def table4(seed, scenes):
    scenes = scenes or REPLICA_SCENES
    rows = []
    for key, name, digits in (("psnr", "PSNR", 2), ("ssim", "SSIM", 3), ("lpips", "LPIPS", 3)):
        values = [value(Run(REPLICA, s), key, seed) for s in scenes]
        rows.append([name, fmt(mean(values), digits)] + [fmt(v, digits) for v in values])
    table("Tab. IV - rendering quality of VCS-SLAM on Replica", ["Metric", "Avg."] + [SHORT.get(s, s) for s in scenes], rows,
          "SSIM is MS-SSIM, as in SplaTAM and SGS-SLAM.")


def table5(seed, scenes):
    per_scene_table("Tab. V - semantic mIoU [%] on Replica (per-frame visible-class protocol)", REPLICA,
                    scenes or SEMANTIC_SCENES, "miou_percent", 2, seed)


def table6(seed, scenes):
    scenes = scenes or REPLICA_SCENES
    miou_scenes = [s for s in scenes if s in SEMANTIC_SCENES] or scenes
    rows = []
    for tag, name in (("no_vcsu", "without VCSU module"), ("no_scea", "without SCEA module"),
                      ("no_cauw", "without CAUW module"), ("", "With All modules")):
        rows.append([name] + [fmt(mean([value(Run(REPLICA, s, tag), k, seed) for s in scenes]), d)
                              for k, d in (("depth_l1_cm", 2), ("ate_rmse_cm", 2), ("psnr", 2))]
                    + [fmt(mean([value(Run(REPLICA, s, tag), "miou_percent", seed) for s in miou_scenes]), 2)])
    table("Tab. VI - ablation on Replica", ["Settings", "Depth L1 [cm]", "ATE RMSE [cm]", "PSNR [dB]", "mIoU [%]"],
          rows, f"Depth L1, ATE and PSNR averaged over {', '.join(scenes)}; "
                f"mIoU over {', '.join(miou_scenes)} (protocol of Tab. V).")


def table7(seed, scenes):
    scenes = scenes or REPLICA_SCENES
    rows = []
    for module, metric, key, scale, digits in (("VCSU", "Depth-inconsistent mIoU (%)", "inconsistent_miou", 100, 2),
                                               ("SCEA", "Boundary F1 Score (BF)", "boundary_f1", 1, 2),
                                               ("CAUW", "Error Detection AUROC", "error_auroc", 1, 2)):
        sgs = mean([value(Run(REPLICA_SGS, s), key, seed) for s in scenes])
        ours = mean([value(Run(REPLICA, s), key, seed) for s in scenes])
        sgs, ours = (None if v is None else v * scale for v in (sgs, ours))
        gain = None if sgs is None or ours is None else ours - sgs
        rows.append([module, metric, fmt(sgs, digits), fmt(ours, digits),
                     "-" if gain is None else f"{gain:+.{digits}f}"])
    table("Tab. VII - mechanism analysis on Replica", ["Module", "Targeted Metric", "SGS-SLAM", "Ours", "Improvement"],
          rows, f"Averaged over {', '.join(scenes)}. Depth-inconsistent: |D - D_obs| > 0.05 m; "
                "boundary F1 within 3 px; AUROC of the ray depth variance for detecting misclassified pixels.")


def table8(seed, scenes):
    scenes = scenes or SEMANTIC_SCENES
    rows = []
    for kind, levels in CORRUPTIONS.items():
        for config, method in ((REPLICA_SGS, "SGS-SLAM"), (REPLICA, "VCS-SLAM (Ours)")):
            clean = mean([value(Run(config, s), "miou_percent", seed) for s in scenes])
            noisy = [mean([value(Run(config, s, corruption_tag(kind, level)), "miou_percent", seed) for s in scenes])
                     for level in levels]
            drop = None if clean is None or noisy[-1] is None else clean - noisy[-1]
            rows.append([f"{kind} ({'/'.join(f'{l:g}' for l in levels)})", method, fmt(clean)]
                        + [fmt(v) for v in noisy] + [fmt(drop)])
    table("Tab. VIII - mIoU [%] under corrupted 2D priors", ["Corruption", "Method", "Clean", "Mild", "Mod.", "Severe",
                                                              "Delta"], rows,
          f"Averaged over {', '.join(scenes)}; Delta = Clean - Severe.")


def table9(seed, scenes):
    scenes = scenes or SEMANTIC_SCENES
    kind, level = STRUCTURAL
    struct = corruption_tag(kind, level)
    rows = []
    for tag, name in (("hardgate", "Hard gate"), ("tv", "TV smoothness"), ("entropy", "Entropy weighting"),
                      ("", "VCS-SLAM (Ours)")):
        clean = mean([value(Run(REPLICA, s, tag), "miou_percent", seed) for s in scenes])
        noisy = mean([value(Run(REPLICA, s, "_".join(t for t in (tag, struct) if t)), "miou_percent", seed)
                      for s in scenes])
        drop = None if clean is None or noisy is None else clean - noisy
        rows.append([name, fmt(clean), fmt(noisy), fmt(drop)])
    table("Tab. IX - naive reweighting variants, mIoU [%]", ["Variant", "Clean", "Struct.", "Delta"], rows,
          f"Averaged over {', '.join(scenes)}; Struct. = boundary misalignment at k = {level} px.")


def sensitivity(seed, scenes):
    scene = (scenes or ["room0"])[0]
    reference = value(Run(REPLICA, scene), "miou_percent", seed)
    rows = []
    for param, values in SENSITIVITY.items():
        for v in values:
            tag = "" if v == PAPER_DEFAULTS[param] else f"{param}{v:g}"
            miou = value(Run(REPLICA, scene, tag), "miou_percent", seed)
            change = None if miou is None or reference is None else miou - reference
            rows.append([param, f"{v:g}" + (" (paper)" if not tag else ""), fmt(miou),
                         "-" if change is None else f"{change:+.2f}"])
    table(f"Sensitivity on {scene} (one parameter at a time), mIoU [%]",
          ["Parameter", "Value", "mIoU", "Change"], rows)


TABLES = {"table1": table1, "table2": table2, "table3": table3, "table4": table4, "table5": table5,
          "table6": table6, "table7": table7, "table8": table8, "table9": table9, "sensitivity": sensitivity}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", choices=list(TABLES) + ["all"])
    parser.add_argument("--scenes", nargs="+", help="Scenes to average over (default: those of the paper)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in TABLES if args.table == "all" else [args.table]:
        TABLES[name](args.seed, args.scenes)


if __name__ == "__main__":
    main()
