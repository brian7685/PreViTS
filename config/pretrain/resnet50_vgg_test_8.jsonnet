local base = import "moco-train-base_vggs_test.jsonnet";

base {
    batch_size: 8,
    num_workers: 16,

    arch: 'resnet50',
    spatial_transforms+: {
        size: 224,
    },
}
