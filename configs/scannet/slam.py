"""VCS-SLAM on ScanNet (Tab. II), with the settings of the paper (Sec. IV-A4).

    python scripts/slam.py configs/scannet/slam.py --scene scene0000_00

Data is read from $VCS_DATA_ROOT/scannet (default ./data/scannet), preprocessed with
preprocess/scannet/run.py. Results go to $VCS_OUTPUT_ROOT/ScanNet/<scene>_<seed>[_<tag>].
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("VCS_DATA_ROOT", "./data")
OUTPUT_ROOT = os.environ.get("VCS_OUTPUT_ROOT", "./experiments")

SCENES = ["scene0000_00", "scene0059_00", "scene0106_00", "scene0169_00", "scene0181_00", "scene0207_00"]


def get_config(scene_name="scene0000_00", seed=0, tag=""):
    group_name = "ScanNet"
    run_name = f"{scene_name}_{seed}" + (f"_{tag}" if tag else "")
    tracking_iters = 100
    mapping_iters = 30
    return dict(
        workdir=os.path.join(OUTPUT_ROOT, group_name),
        run_name=run_name,
        seed=seed,
        primary_device="cuda:0",
        map_every=8,  # global mapping and Gaussian densification every N = 8 frames
        keyframe_every=5,
        mapping_window_size=10,
        report_global_progress_every=500,
        eval_every=5,
        scene_radius_depth_ratio=3,
        mean_sq_dist_method="projective",
        report_iter_progress=False,
        load_checkpoint=False,
        checkpoint_time_idx=0,
        save_checkpoints=False,
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
            basedir=os.path.join(DATA_ROOT, "scannet"),
            gradslam_data_cfg=os.path.join(_HERE, "..", "data", "scannet.yaml"),
            sequence=scene_name,
            desired_image_height=480,
            desired_image_width=640,
            start=0,
            end=-1,
            stride=1,
            num_frames=-1,
            load_semantics=True,
            num_semantic_classes=48,
        ),
        semantics=dict(
            representation="logits",
            init_logit=1.0,
            vcsu=True,
            scea=True,
            cauw=True,
            sigma_vis=0.05,
            beta=0.7,
            gamma=100.0,
            lambda_edge=0.02,
            tau_d=0.75,
            eps=1e-8,
            vcsu_variant="soft",
            scea_variant="edge",
            cauw_variant="variance",
            prior_corruption=dict(kind=None, level=0.0, seed=0, small_object_frac=0.01),
        ),
        mechanism=dict(inconsistent_depth=0.05, boundary_tolerance=3),
        tracking=dict(
            use_gt_poses=False,
            forward_prop=True,
            num_iters=tracking_iters,
            use_sil_for_loss=True,
            sil_thres=0.99,
            use_l1=True,
            ignore_outlier_depth_loss=False,
            loss_weights=dict(im=0.5, depth=1.0, seg=0.0),
            lrs=dict(
                means3D=0.0,
                rgb_colors=0.0,
                unnorm_rotations=0.0,
                logit_opacities=0.0,
                log_scales=0.0,
                cam_unnorm_rots=0.0005,
                cam_trans=0.0005,
                semantic_logits=0.0,
            ),
        ),
        mapping=dict(
            num_iters=mapping_iters,
            add_new_gaussians=True,
            sil_thres=0.5,
            use_l1=True,
            use_sil_for_loss=False,
            ignore_outlier_depth_loss=False,
            loss_weights=dict(im=0.5, depth=1.0, seg=0.01),
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
            render_mode="color",
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
