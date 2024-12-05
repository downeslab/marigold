import implementation


def run_figure_4(seeds):
    # Touch-evoked
    for seed in seeds:
        # Hierarchical
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "hierarchical",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "batch",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

        # Isotropic
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "batch",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

    # Visuomotor
    for seed in seeds:
        # Hierarchical
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "hierarchical",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "batch",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )

        # Isotropic
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "batch",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )


def run_figure_5(seeds):
    # Touch-evoked
    for seed in seeds:
        # MobileNetV3
        # This run commented out because it's already included in `run_figure_4`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "touch-evoked",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 32,
        #     norm = "batch",
        #     activation = "hard-swish",
        #     expansion_norm = True,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = True,
        #     attention = None,
        #     projection_norm = True,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 5.0e-4,
        #     num_epochs = 250
        # )

        # "Swap"
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

        # "Drop"
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = False,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

    # Visuomotor
    for seed in seeds:
        # MobileNetV3
        # This run commented out because it's already included in `run_figure_4`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "visuomotor",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 16,
        #     norm = "batch",
        #     activation = "hard-swish",
        #     expansion_norm = True,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = True,
        #     attention = None,
        #     projection_norm = True,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 1.0e-4,
        #     num_epochs = 500
        # )

        # "Swap"
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )

        # "Drop"
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = False,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )


def run_figure_6(seeds):
    # Touch-evoked
    for seed in seeds:
        # 384 images
        # This run commented out because it's already included in `run_figure_5`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "touch-evoked",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 32,
        #     norm = "instance",
        #     activation = "hard-swish",
        #     expansion_norm = False,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = False,
        #     attention = None,
        #     projection_norm = False,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 5.0e-4,
        #     num_epochs = 250
        # )

        # 192 images
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = "half",
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = False,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 500
        )

        # 96 images
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = "three-quarters",
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = False,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 1000
        )

    # Visuomotor
    for seed in seeds:
        # 192 images
        # This run commented out because it's already included in `run_figure_5`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "visuomotor",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 16,
        #     norm = "instance",
        #     activation = "hard-swish",
        #     expansion_norm = False,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = False,
        #     attention = None,
        #     projection_norm = False,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 1.0e-4,
        #     num_epochs = 500
        # )

        # 96 images
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = "half",
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = False,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 1000
        )

        # 48 images
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = "three-quarters",
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = False,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 2000
        )


def run_supplementary_figure_1(seeds):
    # Touch-evoked
    for seed in seeds:
        # 4x4 patch size
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 4,
            num_blocks = 10,
            num_features = 8,
            norm = "batch",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

        # 8x8 patch size
        # This run commented out because it's already included in `run_figure_4`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "touch-evoked",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 32,
        #     norm = "batch",
        #     activation = "hard-swish",
        #     expansion_norm = True,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = True,
        #     attention = None,
        #     projection_norm = True,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 5.0e-4,
        #     num_epochs = 250
        # )

        # 16x16 patch size
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 16,
            num_blocks = 10,
            num_features = 128,
            norm = "batch",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

    # Visuomotor
    for seed in seeds:
        # 4x4 patch size
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 4,
            num_blocks = 10,
            num_features = 4,
            norm = "batch",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )

        # 8x8 patch size
        # This run commented out because it's already included in `run_figure_4`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "visuomotor",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 16,
        #     norm = "batch",
        #     activation = "hard-swish",
        #     expansion_norm = True,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = True,
        #     attention = None,
        #     projection_norm = True,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 1.0e-4,
        #     num_epochs = 500
        # )

        # 16x16 patch size
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 16,
            num_blocks = 10,
            num_features = 64,
            norm = "batch",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )


def run_supplementary_figure_2(seeds):
    # Touch-evoked
    for seed in seeds:
        # Batch Norm
        # This run commented out because it's already included in `run_figure_4`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "touch-evoked",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 32,
        #     norm = "batch",
        #     activation = "hard-swish",
        #     expansion_norm = True,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = True,
        #     attention = None,
        #     projection_norm = True,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 5.0e-4,
        #     num_epochs = 250
        # )

        # Layer Norm
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "layer",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

        # Group Norm
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "group",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

        # Instance Norm
        # This run commented out because it's already included in `run_figure_5`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "touch-evoked",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 32,
        #     norm = "instance",
        #     activation = "hard-swish",
        #     expansion_norm = True,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = True,
        #     attention = None,
        #     projection_norm = True,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 5.0e-4,
        #     num_epochs = 250
        # )

    # Visuomotor
    for seed in seeds:
        # Batch Norm
        # This run commented out because it's already included in `run_figure_4`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "visuomotor",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 16,
        #     norm = "batch",
        #     activation = "hard-swish",
        #     expansion_norm = True,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = True,
        #     attention = None,
        #     projection_norm = True,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 1.0e-4,
        #     num_epochs = 500
        # )

        # Layer Norm
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "layer",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )

        # Group Norm
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "group",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )

        # Instance Norm
        # This run commented out because it's already included in `run_figure_5`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "visuomotor",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 16,
        #     norm = "instance",
        #     activation = "hard-swish",
        #     expansion_norm = True,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = True,
        #     attention = None,
        #     projection_norm = True,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 1.0e-4,
        #     num_epochs = 500
        # )


def run_supplementary_figure_3(seeds):
    # Touch-evoked
    for seed in seeds:
        # Instance Norm 1, Hard Swish 1
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = False,
            depthwise_activation = False,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

        # Instance Norm 1, Hard Swish 2
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = False,
            depthwise_filter_size = 5,
            depthwise_norm = False,
            depthwise_activation = True,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

        # Instance Norm 2, Hard Swish 1
        # This run commented out because it's already included in `run_figure_5`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "touch-evoked",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 32,
        #     norm = "instance",
        #     activation = "hard-swish",
        #     expansion_norm = False,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = False,
        #     attention = None,
        #     projection_norm = False,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 5.0e-4,
        #     num_epochs = 250
        # )

        # Instance Norm 2, Hard Swish 2
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = False,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

        # Instance Norm 3, Hard Swish 1
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = False,
            depthwise_activation = False,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

        # Instance Norm 3, Hard Swish 2
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "touch-evoked",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 32,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = False,
            depthwise_filter_size = 5,
            depthwise_norm = False,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 5.0e-4,
            num_epochs = 250
        )

    # Visuomotor
    for seed in seeds:
        # Instance Norm 1, Hard Swish 1
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = False,
            depthwise_activation = False,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )

        # Instance Norm 1, Hard Swish 2
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = True,
            expansion_activation = False,
            depthwise_filter_size = 5,
            depthwise_norm = False,
            depthwise_activation = True,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )

        # Instance Norm 2, Hard Swish 1
        # This run commented out because it's already included in `run_figure_5`
        # model, model_ema = implementation.train_model(
        #     seed = seed,
        #     dataset = "visuomotor",
        #     training_data_ablation = None,
        #     architecture = "isotropic",
        #     patch_size = 8,
        #     num_blocks = 10,
        #     num_features = 16,
        #     norm = "instance",
        #     activation = "hard-swish",
        #     expansion_norm = False,
        #     expansion_activation = True,
        #     depthwise_filter_size = 5,
        #     depthwise_norm = True,
        #     depthwise_activation = False,
        #     attention = None,
        #     projection_norm = False,
        #     training_batch_size = 16,
        #     gradient_accumulation_size = 16,
        #     evaluation_batch_size = 16,
        #     learning_rate = 1.0e-4,
        #     num_epochs = 500
        # )

        # Instance Norm 2, Hard Swish 2
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = False,
            depthwise_filter_size = 5,
            depthwise_norm = True,
            depthwise_activation = True,
            attention = None,
            projection_norm = False,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )

        # Instance Norm 3, Hard Swish 1
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = True,
            depthwise_filter_size = 5,
            depthwise_norm = False,
            depthwise_activation = False,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )

        # Instance Norm 3, Hard Swish 2
        model, model_ema = implementation.train_model(
            seed = seed,
            dataset = "visuomotor",
            training_data_ablation = None,
            architecture = "isotropic",
            patch_size = 8,
            num_blocks = 10,
            num_features = 16,
            norm = "instance",
            activation = "hard-swish",
            expansion_norm = False,
            expansion_activation = False,
            depthwise_filter_size = 5,
            depthwise_norm = False,
            depthwise_activation = True,
            attention = None,
            projection_norm = True,
            training_batch_size = 16,
            gradient_accumulation_size = 16,
            evaluation_batch_size = 16,
            learning_rate = 1.0e-4,
            num_epochs = 500
        )


if __name__ == "__main__":
    seeds = range(10)

    run_figure_4(seeds)
    run_figure_5(seeds)
    run_figure_6(seeds)

    run_supplementary_figure_1(seeds)
    run_supplementary_figure_2(seeds)
    run_supplementary_figure_3(seeds)
