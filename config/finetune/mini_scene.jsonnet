local mini_scene = import '../dataset/mini_scene.libsonnet';
local s3dg = import '../model/s3dg.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: mini_scene,
    model: s3dg,
    model_type: 'multitask',
    spatial_transforms+: {
        size: 224
    },
    temporal_transforms+: {
        size: 64
    },
    batch_size: 4,
    validate: {
        batch_size: 8,
    },
    final_validate: {
        batch_size: 2,
    },
    optimizer+: {lr: 0.005},
    num_epochs: 50
}
