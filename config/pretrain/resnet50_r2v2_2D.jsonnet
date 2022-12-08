local base = import "moco-train-base_r2v2.jsonnet";
local sgd = import "../optimizer/sgd03.libsonnet";

base {
    batch_size: 16,
    num_workers: 4,

    spatial_transforms+: {
        size: 224,
    },
    arch: 'resnet50_2D', //'resnet50_2D'
    optimizer: sgd,
    

    model: {
        arch: $.arch,
    },
    moco: {
        dim: 128,
        k: 16384,
        m: 0.999,
        t: 0.07,
        mlp: false,
        diff_speed: [], // Avalable choices: [2] (2x speed)，[4] (4x speed), [4,2,1] (randomly choose a speed)，[] (not enabled)
        aug_plus: false,
        fc_type: 'linear', // Avalable choices: linear, mlp, conv
    },

    temporal_transforms: {
        _size:: 1,
        size: if std.length($.moco.diff_speed) == 0 then self._size else $.moco.diff_speed[0] * self._size,
        strides: [
            {stride: 1, weight: 1},
        ], 
        frame_rate: null, // Follow the origin video fps if not set. Use fixed fps if set.
        random_crop: true,
    },
}
