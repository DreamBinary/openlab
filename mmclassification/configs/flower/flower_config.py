_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

model = dict(
    head=dict(
        num_classes=5,
        topk=(1,)
    ),
)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        data_prefix='/HOME/scz5161/run/ssh/mmclassification/data/',
        ann_file='/HOME/scz5161/run/ssh/mmclassification/data/flower/train.txt',
        classes='/HOME/scz5161/run/ssh/mmclassification/data/flower/classes.txt'
    ),

    val=dict(
        data_prefix='/HOME/scz5161/run/ssh/mmclassification/data/',
        ann_file='/HOME/scz5161/run/ssh/mmclassification/data/flower/val.txt',
        classes='/HOME/scz5161/run/ssh/mmclassification/data/flower/classes.txt'
    )
)
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, )})
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3, 6, 9], gamma=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
# 预训练模型
load_from = "/HOME/scz5161/run/ssh/mmclassification/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
work_dir = '/HOME/scz5161/run/ssh/mmclassification/work_dir'  # 用于保存当前实验的模型检查点和日志的目录文件地址。
