_base_ = [
    "mask_rcnn_r50_fpn_2x_coco.py"
]

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='mmdetection/data/balloon/annotations/train.json',
        img_prefix='mmdetection/data/balloon/train/',
    ),
    val=dict(
        type='CocoDataset',
        ann_file='mmdetection/data/balloon/annotations/val.json',
        img_prefix='mmdetection/data/balloon/val/',
    ),
    test=dict(
        type='CocoDataset',
        ann_file='mmdetection/data/balloon/annotations/val.json',
        img_prefix='mmdetection/data/balloon/val/',
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
load_from = "mmdetection/checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth"
work_dir = "mmdetection/work_dir"