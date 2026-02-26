from pathlib import Path

from groot.vla.data.schema import EmbodimentTag

EMBODIMENT_TAGS_TO_DATASET_PATHS = {
    # LeRobot official datasets
    EmbodimentTag.GR1_UNIFIED_SEGMENTATION: list(
        Path("/mnt/amlfs-03/shared/datasets/segmentation_78dc_1_episode_2").glob("*"),
    ),
    EmbodimentTag.DREAM: [
        Path(
            "/mnt/amlfs-01/home/seonghyeony/data/dreams_lerobot/dream.dream_real_point_lerobot_0305_idm_nostate_60k"
        ),
    ],
    EmbodimentTag.LAPA: [
        Path("/mnt/amlfs-03/shared/datasets/lerobot_lapa/dreams/somethingv2/lapa.split_1")
    ]
    + list(
        Path(
            "/mnt/amlfs-03/shared/datasets/lerobot_lapa/agibot/agibot_set01_conversion/agibotworld"
        ).glob("task*")
    )
    + list(
        Path(
            "/mnt/amlfs-03/shared/datasets/lerobot_lapa/agibot/agibot_set03_conversion/agibotworld"
        ).glob("task*")
    )
    + list(
        Path(
            "/mnt/amlfs-03/shared/datasets/lerobot_lapa/agibot/agibot_set04_conversion/agibotworld"
        ).glob("task*")
    )
    + list(
        Path(
            "/mnt/amlfs-03/shared/datasets/lerobot_lapa/agibot/agibot_set05_conversion/agibotworld"
        ).glob("task*")
    )
    + list(
        Path(
            "/mnt/amlfs-03/shared/datasets/lerobot_lapa/agibot/agibot_set06_conversion/agibotworld"
        ).glob("task*")
    ),
    EmbodimentTag.OXE_DROID: [
        Path("/mnt/amlfs-03/shared/datasets/lerobot/OXE/oxe_droid.droid_success_only_pad_res256"),
    ],
    EmbodimentTag.OXE_FRACTAL: [
        Path("/mnt/amlfs-03/shared/datasets/lerobot/OXE/oxe_fractal.fractal_success_only_v2"),
    ],
    EmbodimentTag.OXE_LANGUAGE_TABLE: [
        Path("/mnt/amlfs-03/shared/datasets/lerobot/OXE/oxe_language_table.table_success_only"),
    ],
    EmbodimentTag.OXE_BRIDGE: [
        Path("/mnt/amlfs-03/shared/datasets/lerobot/OXE/oxe_bridge.bridge_success_only_v2"),
    ],
    EmbodimentTag.OXE_MUTEX: {
        Path("/mnt/amlfs-03/shared/datasets/lerobot/OXE/oxe_mutex.mutex_success_only")
    },
    EmbodimentTag.OXE_PLEX: {
        Path("/mnt/amlfs-03/shared/datasets/lerobot/OXE/oxe_plex.plex_success_only")
    },
    EmbodimentTag.OXE_ROBOSET: {
        Path("/mnt/amlfs-03/shared/datasets/lerobot/OXE/oxe_roboset.roboset_success_only")
    },
    EmbodimentTag.LANGUAGE_TABLE_SIM: [
        Path("/mnt/amlfs-03/shared/datasets/lerobot/language_table_sim.full"),
    ],
    # 7DC datasets
    EmbodimentTag.ROBOCASA_GR1_ARMS_ONLY_FOURIER_HANDS: list(
        Path("/mnt/amlfs-03/shared/datasets/lerobot/7DC").glob(
            "robocasa_gr1_arms_only_fourier_hands.*"
        )
    )
    + list(
        Path("/mnt/amlfs-03/shared/datasets/lerobot/9W").glob(
            "robocasa_gr1_arms_only_fourier_hands.*"
        )
    ),
    # 24DC datasets
    EmbodimentTag.ROBOCASA_GR1_ARMS_WAIST_FOURIER_HANDS: list(
        Path("/mnt/amlfs-03/shared/datasets/lerobot/24DC_armsandwaist").glob(
            "robocasa_gr1_arms_waist_fourier_hands.*"
        )
    ),
    # 24P datasets
    EmbodimentTag.ROBOCASA_PANDA_OMRON: list(
        Path("/mnt/amlfs-03/shared/datasets/lerobot/24P").glob("robocasa_panda_omron.*")
    ),
    # The rest of 9W datasets
    EmbodimentTag.ROBOCASA_BIMANUAL_PANDA_INSPIRE_HAND: list(
        Path("/mnt/amlfs-03/shared/datasets/lerobot/9W").glob(
            "robocasa_bimanual_panda_inspire_hand.*"
        )
    ),
    EmbodimentTag.ROBOCASA_BIMANUAL_PANDA_PARALLEL_GRIPPER: list(
        Path("/mnt/amlfs-03/shared/datasets/lerobot/9W").glob(
            "robocasa_bimanual_panda_parallel_gripper.*"
        )
    ),
    EmbodimentTag.ROBOCASA_GR1_FIXED_LOWER_BODY_FOURIER_HANDS: list(
        Path("/mnt/amlfs-03/shared/datasets/lerobot/9W").glob(
            "robocasa_gr1_fixed_lower_body_fourier_hands.*"
        )
    ),
    # Real GR1 arms waist annotated
    EmbodimentTag.REAL_GR1_ARMS_WAIST_ANNOTATED: list(
        Path("/mnt/amlfs-03/shared/datasets/lerobot/1RA").glob("real_gr1_arms_waist_annotated.*")
    ),
    # Real GR1 arms only annotated
    # version: 0217
    # EmbodimentTag.REAL_GR1_ARMS_ONLY_ANNOTATED: list(
    #     Path("/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1_lerobot").glob(
    #         "real_gr1_arms_only_annotated.*"
    #     )
    # ),
    # version: 0224_cotrain_simreal
    # EmbodimentTag.REAL_GR1_ARMS_ONLY_ANNOTATED: list(
    #     Path("/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1_remove_static_lerobot").glob(
    #         "real_gr1_arms_only_annotated.*CoTrain"
    #     )
    # )
    # + list(
    #     Path("/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1_new_5DC-S_lerobot").glob(
    #         "real_gr1_arms_only_annotated.*CoTrain"
    #     )
    # ),
    # version: 0224_cotrain_oldsimreal
    EmbodimentTag.REAL_GR1_ARMS_ONLY_ANNOTATED: list(
        Path("/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1_remove_static_lerobot").glob(
            "real_gr1_arms_only_annotated.*CoTrain"
        )
    )
    + list(
        Path("/mnt/amlfs-03/shared/datasets/gr1/real_gr1_arms_only_v1_old_5DC-S_lerobot").glob(
            "real_gr1_arms_only_annotated.*CoTrain"
        )
    ),
    # Human dataset
    EmbodimentTag.HOT3D_HANDS_ONLY: list(
        Path("/mnt/amlfs-02/shared/datasets/hot3d/lerobot_data").glob("hot3d_hands_only.*")
    ),
    EmbodimentTag.GR1_UNIFIED: [
        # RU-0226
        Path(
            "/mnt/amlfs-03/shared/datasets/lerobot/GR1/2RA_coarse_fine_0226/gr1_unified.UnzeroedArmsOnlyRemoveStaticSliceBGCropPad256Freq20"
        ),
        Path(
            "/mnt/amlfs-03/shared/datasets/lerobot/GR1/2RA_coarse_fine_0226/gr1_unified.UnzeroedArmsWaistRemoveStaticSliceBGCropPad256Freq20"
        ),
        # 5DC-R-unlocked
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/5DC-R-unlocked/gr1_unified.Slice_arms_waist_5DC-R-unlocked_cutting_board_to_basket_dummylang_remove_static_threshold1_BGCropRes256Freq20"
        ),
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/5DC-R-unlocked/gr1_unified.Slice_arms_waist_5DC-R-unlocked_cutting_board_to_pan_dummylang_remove_static_threshold1_BGCropRes256Freq20"
        ),
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/5DC-R-unlocked/gr1_unified.Slice_arms_waist_5DC-R-unlocked_placemat_to_basket_dummylang_remove_static_threshold1_BGCropRes256Freq20"
        ),
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/5DC-R-unlocked/gr1_unified.Slice_arms_waist_5DC-R-unlocked_plate_to_bowl_dummylang_remove_static_threshold1_BGCropRes256Freq20"
        ),
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/5DC-R-unlocked/gr1_unified.Slice_arms_waist_5DC-R-unlocked_tray_to_plate_dummylang_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Pointing
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0302/gr1_unified.pointing_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # PnP
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0302/gr1_unified.pnp_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Industrial 0303
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0303/gr1_unified.industrial_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Cylinder 0304
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0304/gr1_unified.industrial_cylinder_to_bin_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Pour 0304
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0304/gr1_unified.industrial_pour_and_put_on_black_platform_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # GTC-coordination-part1 0304 + 0305
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0305/gr1_unified.gtc_coordination_part1_remove_static_both_threshold1_BGCropRes256Freq20"
        ),
        # GTC-coordination-part2 0304
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0304/gr1_unified.gtc_coordination_part2_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Sort 0304
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0304/gr1_unified.industrial_sort_dummylang_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Articulated
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0305/gr1_unified.articulated_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Cylinder 0305
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0305/gr1_unified.industrial_cylinder_to_bin_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Pour 0305
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0305/gr1_unified.industrial_pour_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Industrial 0306
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0306/gr1_unified.industrial_objects_to_bin_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Pour 0306
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0306/gr1_unified.pour_to_clear_bin_remove_static_threshold1_BGCropRes256Freq20"
        ),
        # Hand over cylinder 0306
        Path(
            "/mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0306/gr1_unified.hand_over_cylinder_to_bin_remove_static_threshold1_BGCropRes256Freq20"
        ),
    ]
    + list(
        Path("/mnt/amlfs-03/shared/datasets/lerobot/78dc_fixed_annotations").glob("gr1_unified.*")
    )[::10],
    EmbodimentTag.AGIBOT: [
        Path(
            "/mnt/amlfs-03/shared/datasets/agibot-dbg/agibot-beta-converted/375/agibotworld/task_375"
        )
    ],
    EmbodimentTag.GR1_ISAAC: [
        Path(
            "/mnt/amlfs-03/shared/datasets/shiwei/exhaust_pipe_dataset_generated_1000_human_demos_5_v10_noise_003/lerobot"
        ),
        Path(
            "/mnt/amlfs-03/shared/datasets/shiwei/nut_pouring_dataset_generated_1000_human_demos_5_v34_action_noise_003/lerobot"
        ),
    ],
    EmbodimentTag.UNITREE_G1_FULL_BODY_WITH_HEIGHT: [
        Path(
            "/mnt/amlfs-02/shared/datasets/oss_testing/G1/unitree_g1_full_body_with_height.g1_pickup_box"
        ),
    ],
    EmbodimentTag.UNITREE_G1_FULL_BODY_WITH_HEIGHT_AND_EEF: [
        Path(
            "/mnt/amlfs-02/shared/datasets/oss_testing/G1/g1_full_body_height_eef.g1_pickup_bottle"
        ),
    ],
    # LIBERO sim - paths configured via CLI libero_data_root
    EmbodimentTag.LIBERO_SIM: [],
}

DATASET_PATHS_TO_EMBODIMENT_TAGS = {
    path: dataset_tag
    for dataset_tag, dataset_paths in EMBODIMENT_TAGS_TO_DATASET_PATHS.items()
    for path in dataset_paths
}
