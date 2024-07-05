# Modify training settings to be the same with UACMDet.
# model settings
angle_version = 'le90'
model = dict(
    type='BimodalOrientedRCNN',
    backbone1=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        strides=(1, 2, 2, 1),
        dilations=(1, 1, 1, 2),
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint="/data2/likaihao/mmcv_imagenet_models/resnet50-19c8e357.pth")),
    backbone2=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        strides=(1, 2, 2, 1),
        dilations=(1, 1, 1, 2),
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint="/data2/likaihao/mmcv_imagenet_models/resnet50-19c8e357.pth")),
    fusion_neck=dict(
        type='DATFFFusionNeck',
        in_channels=[2048],
        seq_length=256,
        grids=10,
        num_heads=16,
        num_layers1=1,
        num_layers2=1,
        mlp_ratio=4,
        proj_channel=256,
        no_layer_norm=False,
        no_out_layer_norm=False,
        no_poem=True,
        poem_drop=0.,
        attn_drop=0.,
        mlp_drop=0.,
        poem_type='sinusoidal',
        out_type='cat-conv',
        grid_size=[8]),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=2048,
        feat_channels=2048,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1, 2, 4, 8, 16],
            ratios=[1.0/3.0, 1.0/2.0, 1.0, 2.0, 3.0],
            strides=[16]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[16]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=2048,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=5,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            multiclass=True,
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))
# dataset settings
dataset_type = 'BimodalRDroneVehicleDataset'
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