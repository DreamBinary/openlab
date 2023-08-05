_base_ = [
    '../_base_/models/resnet50_cifar.py', '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        data_prefix='/HOME/scz5161/run/ssh/mmclassification/data/cifar10',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        data_prefix='/HOME/scz5161/run/ssh/mmclassification/data/cifar10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        data_prefix='/HOME/scz5161/run/ssh/mmclassification/data/cifar10',
        pipeline=[
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, )})
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3, 6, 9], gamma=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
# 预训练模型
load_from = "/HOME/scz5161/run/ssh/mmclassification/checkpoints/resnet50_b16x8_cifar10_20210528-f54bfad9.pth"
work_dir = '/HOME/scz5161/run/ssh/mmclassification/work_dir/cifar10'  # 用于保存当前实验的模型检查点和日志的目录文件地址。
