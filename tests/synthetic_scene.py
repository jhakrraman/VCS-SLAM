"""A tiny ray-cast RGB-D sequence with semantic labels, written in SGS-SLAM's Replica layout (tests only).

The scene: a textured back wall with a coplanar poster and an unlabeled patch, a floor, a box in
front of the wall (depth discontinuity) and a small object on the floor. The camera translates
slowly; its orientation stays fixed, so camera and world axes are aligned (OpenCV convention).
"""

import os

import imageio
import numpy as np

WIDTH, HEIGHT = 40, 30
FX = FY = 32.0
CX, CY = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0
DEPTH_SCALE = 6553.5

# class id -> semantic palette colour
PALETTE = {0: (0, 0, 0), 3: (192, 0, 128), 7: (0, 192, 0), 12: (128, 0, 192), 20: (224, 192, 0), 25: (32, 128, 64)}
ALBEDO = {0: (0.5, 0.5, 0.5), 3: (0.7, 0.6, 0.5), 7: (0.3, 0.3, 0.35), 12: (0.8, 0.2, 0.2), 20: (0.2, 0.4, 0.9),
          25: (0.9, 0.9, 0.1)}

DATA_CONFIG = """dataset_name: 'replica'
camera_params:
  image_height: {h}
  image_width: {w}
  fx: {fx}
  fy: {fy}
  cx: {cx}
  cy: {cy}
  png_depth_scale: {s}
  crop_edge: 0
"""


def _box_hit(origin, dirs, lo, hi):
    inv = 1.0 / np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    t0, t1 = (np.asarray(lo) - origin) * inv, (np.asarray(hi) - origin) * inv
    t_near = np.minimum(t0, t1).max(axis=-1)
    t_far = np.maximum(t0, t1).min(axis=-1)
    return np.where((t_near <= t_far) & (t_near > 0), t_near, np.inf)


def render_frame(cam_pos):
    """Return RGB (H, W, 3) uint8, depth (H, W) metres, class ids (H, W) for a camera at ``cam_pos``."""
    origin = np.asarray(cam_pos, dtype=np.float64)
    u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    dirs = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)

    t = np.full((HEIGHT, WIDTH), np.inf)
    label = np.zeros((HEIGHT, WIDTH), dtype=np.int64)

    t_wall = (3.0 - origin[2]) / dirs[..., 2]
    hit = t_wall[..., None] * dirs + origin
    wall_label = np.full_like(label, 3)
    wall_label[(hit[..., 0] > 0.2) & (hit[..., 0] < 0.8) & (hit[..., 1] > -0.6) & (hit[..., 1] < -0.1)] = 20
    wall_label[(hit[..., 0] < -0.7) & (hit[..., 1] < -0.5)] = 0
    t, label = t_wall, wall_label

    t_floor = np.where(dirs[..., 1] > 1e-6, (0.8 - origin[1]) / np.maximum(dirs[..., 1], 1e-6), np.inf)
    closer = (t_floor > 0) & (t_floor < t)
    t, label = np.where(closer, t_floor, t), np.where(closer, 7, label)

    for class_id, lo, hi in ((12, (-0.45, 0.05, 1.8), (0.15, 0.8, 2.2)), (25, (0.45, 0.65, 1.4), (0.6, 0.8, 1.5))):
        t_box = _box_hit(origin, dirs, lo, hi)
        closer = t_box < t
        t, label = np.where(closer, t_box, t), np.where(closer, class_id, label)

    points = t[..., None] * dirs + origin
    albedo = np.zeros((HEIGHT, WIDTH, 3))
    for class_id, rgb in ALBEDO.items():
        albedo[label == class_id] = rgb
    texture = 0.5 + 0.5 * np.sin(7.0 * points[..., 0]) * np.cos(5.0 * points[..., 1] + 3.0 * points[..., 2])
    rgb = np.clip(albedo * (0.6 + 0.4 * texture[..., None]), 0, 1)
    return (rgb * 255).astype(np.uint8), t * dirs[..., 2], label


def write_sequence(root, scene="synth0", num_frames=12):
    """Write a Replica-format sequence to ``root/Replica/<scene>``; returns the path of its data yaml."""
    folder = os.path.join(root, "Replica", scene)
    for sub in ("frames", "depths", "semantic_ids", "semantic_colors"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    poses = []
    for i in range(num_frames):
        cam_pos = (0.02 * i, -0.01 * i, 0.03 * i)
        rgb, depth, label = render_frame(cam_pos)
        imageio.imwrite(os.path.join(folder, "frames", f"frame{i:06d}.jpg"), rgb, quality=95)
        imageio.imwrite(os.path.join(folder, "depths", f"depth{i:06d}.png"), (depth * DEPTH_SCALE).astype(np.uint16))
        imageio.imwrite(os.path.join(folder, "semantic_ids", f"semantic_id{i:06d}.png"), label.astype(np.uint8))
        colors = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for class_id, color in PALETTE.items():
            colors[label == class_id] = color
        imageio.imwrite(os.path.join(folder, "semantic_colors", f"semantic_color{i:06d}.png"), colors)
        c2w = np.eye(4)
        c2w[:3, 3] = cam_pos
        poses.append(" ".join(f"{x:.8f}" for x in c2w.reshape(-1)))
    with open(os.path.join(folder, "traj.txt"), "w") as f:
        f.write("\n".join(poses) + "\n")
    yaml_path = os.path.join(root, "synthetic_replica.yaml")
    with open(yaml_path, "w") as f:
        f.write(DATA_CONFIG.format(h=HEIGHT, w=WIDTH, fx=FX, fy=FY, cx=CX, cy=CY, s=DEPTH_SCALE))
    return yaml_path


def smoke_overrides(data_root, output_root, yaml_path):
    """``--set`` overrides that run a Replica config on the synthetic sequence on the CPU in seconds."""
    return [
        f"data.basedir={os.path.join(data_root, 'Replica')}",
        f"data.gradslam_data_cfg={yaml_path}",
        f"data.desired_image_height={HEIGHT}",
        f"data.desired_image_width={WIDTH}",
        f"workdir={os.path.join(output_root, 'Replica')}",
        "primary_device=cpu",
        "tracking.num_iters=4",
        "mapping.num_iters=4",
        "eval_every=2",
        "save_checkpoints=false",
        "viz.load_semantics=true",
    ]
