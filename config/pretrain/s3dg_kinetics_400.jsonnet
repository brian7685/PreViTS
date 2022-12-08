local base = import "moco-train-base.jsonnet";

base {
    batch_size: 16,
    num_workers: 4,

    arch: 's3dg',
    num_epochs: '400',

    optimizer+: {
        lr: 0.05
    },
    spatial_transforms+: {
        size: 224,
    },
}
