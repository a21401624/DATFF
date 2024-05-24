# dataset settings
dataset_type = 'FLIRDataset'
data_root = '/data2/likaihao/FLIR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='Resize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_list="/data2/likaihao/FLIR/train.txt",
        ann_file=data_root + 'train/trainlabelrtxt/',
        img_prefix=data_root + 'train/trainimg/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_list="/data2/likaihao/FLIR/val.txt",
        ann_file=data_root + 'val/vallabelrtxt/',
        img_prefix=data_root + 'val/valimg/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_list="/data2/likaihao/FLIR/val.txt",
        ann_file=data_root + 'val/vallabelrtxt/',
        img_prefix=data_root + 'val/valimg/',
        pipeline=test_pipeline))
# evaluation
evaluation = dict(interval=1, metric='mAP', save_best='auto', rule='greater')