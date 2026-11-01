"""VCS-SLAM on Replica, with the settings of the paper (Sec. IV-A4).

    python scripts/slam.py configs/replica/slam.py --scene room0

Data is read from $VCS_DATA_ROOT/Replica (default ./data/Replica) and results are written to
$VCS_OUTPUT_ROOT/Replica/<scene>_<seed>[_<tag>] (default ./experiments).
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("VCS_DATA_ROOT", "./data")
OUTPUT_ROOT = os.environ.get("VCS_OUTPUT_ROOT", "./experiments")

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
SEMANTIC_SCENES = ["room0", "room1", "room2", "office0"]  # scenes of the semantic benchmark (Tab. V)


def get_config(scene_name="room0", seed=0, tag=""):
    group_name = "Replica"
    run_name = f"{scene_name}_{seed}" + (f"_{tag}" if tag else "")
    tracking_iters = 40
    mapping_iters = 60
    return dict(
        workdir=os.path.join(OUTPUT_ROOT, group_name),
        run_name=run_name,
        seed=seed,
        primary_device="cuda:0",
        map_every=8,  # global mapping and Gaussian densification every N = 8 frames
        keyframe_every=5,
        mapping_window_size=24,
        report_global_progress_every=500,
        eval_every=5,  # evaluate every 5th frame after SLAM
        scene_radius_depth_ratio=3,  # max first-frame depth to scene radius ratio (pruning / densification)
        mean_sq_dist_method="projective",
        report_iter_progress=False,
        load_checkpoint=False,
        checkpoint_time_idx=0,
        save_checkpoints=True,
        checkpoint_interval=500,
        save_timestamp_keyframes=False,
        use_wandb=False,
        wandb=dict(
            entity=os.environ.get("WANDB_ENTITY"),
            project="VCS-SLAM",
            group=group_name,
            name=run_name,
            save_qual=False,
            eval_save_qual=True,
        ),
        data=dict(
            basedir=os.path.join(DATA_ROOT, "Replica"),
            gradslam_data_cfg=os.path.join(_HERE, "..", "data", "replica.yaml"),
            sequence=scene_name,
            desired_image_height=680,
            desired_image_width=1200,
            start=0,
            end=-1,
            stride=1,
            num_frames=-1,
            load_semantics=True,
            num_semantic_classes=101,
        ),
        semantics=dict(
            representation="logits",  # C-dimensional semantic logits per Gaussian, cross-entropy (Sec. III-A)
            init_logit=1.0,           # a new Gaussian starts at init_logit * one-hot(label of its pixel)
            vcsu=True,                # visibility-consistent semantic updates (Sec. III-C1)
            scea=True,                # surface-coupled edge alignment (Sec. III-C2)
            cauw=True,                # conflict-aware uncertainty weighting (Sec. III-C3)
            sigma_vis=0.05,           # Eq. 5
            beta=0.7,                 # Eq. 11
            gamma=100.0,              # Eq. 10
            lambda_edge=0.02,         # Eq. 12
            tau_d=0.75,               # Eq. 8
            eps=1e-8,                 # Eqs. 7, 9
            vcsu_variant="soft",      # Tab. IX: "hard"
            scea_variant="edge",      # Tab. IX: "tv"
            cauw_variant="variance",  # Tab. IX: "entropy"
            # Structured corruption of the 2D priors (Tab. VIII): kind in {boundary, confusion, missed, uniform}
            prior_corruption=dict(kind=None, level=0.0, seed=0, small_object_frac=0.01),
        ),
        mechanism=dict(  # Tab. VII metrics, computed during the final evaluation
            inconsistent_depth=0.05,  # |D_p - D_p^obs| > 0.05 m marks a depth-inconsistent pixel
            boundary_tolerance=3,     # boundary F1 tolerance band [px]
        ),
        tracking=dict(
            use_gt_poses=False,
            forward_prop=True,
            num_iters=tracking_iters,
            use_sil_for_loss=True,
            sil_thres=0.99,
            use_l1=True,
            ignore_outlier_depth_loss=False,
            loss_weights=dict(
                im=0.5,     # lambda_C^track
                depth=1.0,  # lambda_D^track
                seg=0.0,    # tracking uses RGB-D only
            ),
            lrs=dict(
                means3D=0.0,
                rgb_colors=0.0,
                unnorm_rotations=0.0,
                logit_opacities=0.0,
                log_scales=0.0,
                cam_unnorm_rots=0.0004,
                cam_trans=0.002,
                semantic_logits=0.0,
            ),
        ),
        mapping=dict(
            num_iters=mapping_iters,
            add_new_gaussians=True,
            sil_thres=0.5,  # for adding new Gaussians
            use_l1=True,
            use_sil_for_loss=False,
            ignore_outlier_depth_loss=False,
            loss_weights=dict(
                im=0.5,     # lambda_C^map
                depth=1.0,  # lambda_D^map
                seg=0.01,   # lambda_S, weights L_sem^ours (Eq. 13)
            ),
            lrs=dict(
                means3D=0.0001,
                rgb_colors=0.0025,
                unnorm_rotations=0.001,
                logit_opacities=0.05,
                log_scales=0.001,
                cam_unnorm_rots=0.0000,
                cam_trans=0.0000,
                semantic_logits=0.0025,
            ),
            prune_gaussians=True,
            pruning_dict=dict(
                start_after=0,
                remove_big_after=0,
                stop_after=20,
                prune_every=20,
                removal_opacity_threshold=0.005,
                final_removal_opacity_threshold=0.005,
                reset_opacities=False,
                reset_opacities_every=500,
            ),
            use_gaussian_splatting_densification=False,
            densify_dict=dict(
                start_after=500,
                remove_big_after=3000,
                stop_after=5000,
                densify_every=100,
                grad_thresh=0.0002,
                num_to_split_into=2,
                removal_opacity_threshold=0.005,
                final_removal_opacity_threshold=0.005,
                reset_opacities_every=3000,
            ),
        ),
        viz=dict(
            render_mode="color",  # ["color", "depth", "centers", "semantic_color"]
            offset_first_viz_cam=True,
            show_sil=False,
            visualize_cams=False,
            viz_w=600, viz_h=340,
            viz_near=0.01, viz_far=100.0,
            view_scale=2,
            viz_fps=5,
            enter_interactive_post_online=True,
            scene_name=scene_name,
            load_semantics=True,
        ),
    )


config = get_config()
