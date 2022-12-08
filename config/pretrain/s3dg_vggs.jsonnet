local base = import "moco-train-base_vggs_test.jsonnet";
 
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
}
