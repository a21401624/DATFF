angle_version = 'oc'
model = dict(
    type='BimodalR3Det',
    backbone1=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint="/data2/likaihao/mmcv_imagenet_models/resnet50-19c8e357.pth")),
    backbone2=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint="/data2/likaihao/mmcv_imagenet_models/resnet50-19c8e357.pth")),
    fusion_neck=dict(
        type='SCATFusionNeck',
        in_channels=[256, 512, 1024, 2048],
        grids_h=10,
        grids_w=10,
        num_heads=16,
        num_layers=1,
        mlp_ratio=4,
        proj_channel=256,
        no_layer_norm=False,
        no_out_layer_norm=False,
        no_poem=True,
        poem_drop=0.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        poem_type='sinusoidal',
        out_type='cat-conv'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=4),
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0/3.0, 1.0/2.0, 1.0, 2.0, 3.0],
            strides=[8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    frm_cfgs=[dict(in_channels=256, featmap_strides=[8, 16, 32, 64])],
    num_refine_stages=1,
    refine_heads=[
        dict(
            type='RotatedRetinaRefineHead',
            num_classes=5,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0))
    ],
    train_cfg=dict(
        s0=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        sr=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.5,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1.0]),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
# dataset settings
dataset_type = 'BimodalDroneVehicleDataset'
data_root = '/data2/likaihao/DroneVehicle/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadPairedImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PairedImagesRResize', img_scale=(640, 512)),
    dict(type='PairedImagesRRandomFlip', flip_ratio=0.5),
    dict(type='PairedImagesNormalize', img_norm_cfg1=img_norm_cfg, img_norm_cfg2=img_norm_cfg),
    dict(type='PairedImagesDefaultFormatBundle'),
    dict(type='PairedImagesCollect', keys=['img1', 'img2', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadPairedImagesFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='PairedImagesRResize'),
            dict(type='PairedImagesNormalize', img_norm_cfg1=img_norm_cfg, img_norm_cfg2=img_norm_cfg),
            dict(type='PairedImagesDefaultFormatBundle'),
            dict(type='PairedImagesCollect', keys=['img1', 'img2'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_list="/data2/likaihao/DroneVehicle/train.txt",
        ann_file=data_root + 'train/trainlabelrtxt/',
        img_prefix1=data_root + 'train/trainimgr/',
        img_prefix2=data_root + 'train/trainimg/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_list="/data2/likaihao/DroneVehicle/val.txt",
        ann_file=data_root + 'val/vallabelrtxt/',
        img_prefix1=data_root + 'val/valimgr/',
        img_prefix2=data_root + 'val/valimg/',
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        img_list="/data2/likaihao/DroneVehicle/test.txt",
        ann_file=data_root + 'test/testlabelrtxt/',
        img_prefix1=data_root + 'test/testimgr/',
        img_prefix2=data_root + 'test/testimg/',
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False))
# evaluation
evaluation = dict(interval=1, metric='mAP', save_best='auto', rule='greater')
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

