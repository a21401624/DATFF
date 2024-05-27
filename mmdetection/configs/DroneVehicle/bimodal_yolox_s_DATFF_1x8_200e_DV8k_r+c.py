img_scale = (640, 640)  # height, width

# model settings
model = dict(
    type='BimodalYOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone1=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    backbone2=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    fusion_neck=dict(
        type='DATFFFusionNeck',
        in_channels=[128, 256, 512],
        seq_length=128,
        grids=10,
        num_heads=16,
        num_layers1=1,
        num_layers2=1,
        mlp_ratio=4,
        proj_channel=128,
        no_layer_norm=False,
        no_out_layer_norm=False,
        no_poem=True,
        poem_drop=0.,
        attn_drop=0.,
        mlp_drop=0.,
        poem_type='sinusoidal',
        out_type='cat-conv',
        grid_size=[8, 8, 4]),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=4, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
dataset_type = 'BimodalDroneVehicleDataset'
data_root = '/data2/likaihao/DroneVehicle/'

train_pipeline = [
    dict(type='PairedImagesMosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='PairedImagesRandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='PairedImagesRandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='PairedImagesResize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='PairedImagesPad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    # dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PairedImagesDefaultFormatBundle'),
    dict(type='PairedImagesCollect', keys=['img1', 'img2', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadPairedImagesFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='PairedImagesResize'),
            dict(type='PairedImagesDefaultFormatBundle'),
            dict(type='PairedImagesCollect', keys=['img1', 'img2'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            img_list=data_root + 'train8k.txt',
            ann_file=data_root + 'train/trainlabelrtxt/',
            img_prefix1=data_root + 'train/trainimgr/',
            img_prefix2=data_root + 'train/trainimg/',
            pipeline=[
                dict(type='LoadPairedImagesFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False,
        ),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # img_list=data_root + 'val644.txt',
        # ann_file=data_root + 'val/vallabelrtxt_new/',
        # img_prefix1=data_root + 'val/valimgr/',
        # img_prefix2=data_root + 'val/valimg/',
        img_list=data_root + 'test4k.txt',
        ann_file=data_root + 'test/testlabelrtxt/',
        img_prefix1=data_root + 'test/testimgr/',
        img_prefix2=data_root + 'test/testimg/',
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        img_list=data_root + 'test4k.txt',
        ann_file=data_root + 'test/testlabelrtxt/',
        img_prefix1=data_root + 'test/testimgr/',
        img_prefix2=data_root + 'test/testimg/',
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False))

max_epochs = 200
num_last_epochs = 15
interval = 10

evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='mAP',
    rule='greater')

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=3,  # epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=interval)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'