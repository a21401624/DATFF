# dataset settings
dataset_type = 'BimodalFLIRDataset'
data_root = '/data2/likaihao/FLIR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadPairedImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PairedImagesResize', img_scale=(640, 512)),
    dict(type='PairedImagesRandomFlip', flip_ratio=0.5),
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
            dict(type='PairedImagesResize'),
            dict(type='PairedImagesNormalize', img_norm_cfg1=img_norm_cfg, img_norm_cfg2=img_norm_cfg),
            dict(type='PairedImagesDefaultFormatBundle'),
            dict(type='PairedImagesCollect', keys=['img1', 'img2'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_list=data_root + "train.txt",
        ann_file=data_root + 'train/trainlabelrtxt/',
        img_prefix1=data_root + 'train/trainimgr/',
        img_prefix2=data_root + 'train/trainimg/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_list=data_root + "val.txt",
        ann_file=data_root + 'val/vallabelrtxt/',
        img_prefix1=data_root + 'val/valimgr/',
        img_prefix2=data_root + 'val/valimg/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_list=data_root + "val.txt",
        ann_file=data_root + 'val/vallabelrtxt/',
        img_prefix1=data_root + 'val/valimgr/',
        img_prefix2=data_root + 'val/valimg/',
        pipeline=test_pipeline))
# evaluation
evaluation = dict(interval=1, metric='mAP', save_best='auto', rule='greater')