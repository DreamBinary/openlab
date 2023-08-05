_base_ = "pspnet_r50-d8_512x512_20k_voc12aug.py"
data_root = ""
load_from = ""
data = dict(
    train=dict(
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/train.txt', ),
    val=dict(
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt', ),
    test=dict(
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
    )
)


load_from = ""
runner = dict(type='EpochBasedRunner', max_epochs=5)