local base = import "moco-train-base.jsonnet";

base {
    batch_size: 16,
    num_workers: 4,

    arch: 's3dg',

    optimizer+: {
        lr: 0.05
    },
    spatial_transforms+: {
        size: 224,
    },
    moco: {
        dim: 128,
        k: 65536,
        m: 0.999,
        t: 0.07,
        mlp: false,
        diff_speed: [2], // Avalable choices: [2] (2x speed)，[4] (4x speed), [4,2,1] (randomly choose a speed)，[] (not enabled)
        aug_plus: false,
        fc_type: 'linear', // Avalable choices: linear, mlp, conv
    },
}
