"""Compose the qualitative figures from the renderings saved during evaluation (eval/rendered_*).

    python tools/make_figures.py fig3 --frame 1499
    python tools/make_figures.py fig4 --scene room0 --frame 1499 --crop 560 60 820 220 --crop 40 260 300 420

Fig. 3 shows VCS-SLAM's RGB renderings of Room-0, Room-1 and Office-0 next to the input frames; renders of
other methods can be added as extra columns with ``--extra NAME=RUN_DIR`` (a folder with eval/rendered_rgb).
Fig. 4 shows the input frame, the semantic map of the base model without VCSU, SCEA and CAUW (or of the
SGS-SLAM baseline with ``--baseline sgs_slam``), VCS-SLAM's semantic map, and zoomed crops (baseline above,
VCS-SLAM below). Evaluated frames are 0 and every eval_every-th frame after it (e.g. 4, 9, 14, ... for
eval_every = 5). Panels are written to figures/.
"""

import argparse
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import BASE_MODEL, REPLICA, REPLICA_SGS, Run  # noqa: E402
from vcs_slam.config import load_config  # noqa: E402


def frame_path(run_dir, kind, frame):
    path = os.path.join(run_dir, "eval", f"rendered_{kind}", f"gs_{frame:04d}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found (run missing, or frame {frame} was not evaluated)")
    return path


def input_frame(scene, frame):
    data = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), REPLICA),
                       scene)["data"]
    return os.path.join(data["basedir"], scene, "frames", f"frame{frame:06d}.jpg")


def labelled(path, text, crop=None, width=None):
    image = Image.open(path).convert("RGB")
    if crop:
        image = image.crop(crop)
    if width:
        image = image.resize((width, round(image.height * width / image.width)))
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((6, 6), text, fill="black")
    return canvas


def grid(rows):
    width = max(sum(im.width for im in row) for row in rows)
    height = sum(max(im.height for im in row) for row in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        x = 0
        for im in row:
            canvas.paste(im, (x, y))
            x += im.width
        y += max(im.height for im in row)
    return canvas


def fig3(args):
    rows = []
    for scene in args.scenes or ["room0", "room1", "office0"]:
        row = [labelled(frame_path(Run(REPLICA, scene).run_dir(args.seed), "rgb", args.frame), f"VCS-SLAM {scene}",
                        width=args.width)]
        for name, run_dir in (e.split("=", 1) for e in args.extra):
            row.insert(0, labelled(frame_path(run_dir.replace("{scene}", scene), "rgb", args.frame), name,
                                   width=args.width))
        row.append(labelled(input_frame(scene, args.frame), "Ground Truth", width=args.width))
        rows.append(row)
    return grid(rows)


def fig4(args):
    scene = (args.scenes or ["room0"])[0]
    if args.baseline == "sgs_slam":
        baseline_run, name = Run(REPLICA_SGS, scene), "SGS-SLAM"
    else:
        baseline_run, name = Run(REPLICA, scene, BASE_MODEL[0]), "without VCSU, SCEA, CAUW"
    base = frame_path(baseline_run.run_dir(args.seed), "seg", args.frame)
    ours = frame_path(Run(REPLICA, scene).run_dir(args.seed), "seg", args.frame)
    rows = [[labelled(input_frame(scene, args.frame), "(a) input RGB", width=args.width),
             labelled(base, f"(b) {name}", width=args.width),
             labelled(ours, "(c) VCS-SLAM (Ours)", width=args.width)]]
    crops = [tuple(c) for c in args.crop]
    if crops:
        rows.append([labelled(base, f"baseline crop {i + 1}", c, args.width) for i, c in enumerate(crops)])
        rows.append([labelled(ours, f"VCS-SLAM crop {i + 1}", c, args.width) for i, c in enumerate(crops)])
    return grid(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=["fig3", "fig4"])
    parser.add_argument("--frame", type=int, required=True, help="An evaluated frame index")
    parser.add_argument("--scenes", nargs="+", help="Scenes (fig3: rows; fig4: the first one is used)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=480, help="Panel width in pixels")
    parser.add_argument("--crop", type=int, nargs=4, action="append", default=[], metavar=("X0", "Y0", "X1", "Y1"),
                        help="fig4: zoom-in region in pixels of the rendered image (repeatable)")
    parser.add_argument("--baseline", choices=["base", "sgs_slam"], default="base",
                        help="fig4 panel (b): base model without the three modules (paper) or SGS-SLAM")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=RUN_DIR",
                        help="fig3: another method's run folder; {scene} is replaced by the scene name")
    parser.add_argument("--out", default="figures", help="Output folder")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.figure}.png")
    (fig3 if args.figure == "fig3" else fig4)(args).save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
