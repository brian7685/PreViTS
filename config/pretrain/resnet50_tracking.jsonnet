local base = import "moco-train-base_tracking.jsonnet";

base {
    batch_size: 16,
    num_workers: 16,

    arch: 'resnet50',
    spatial_transforms+: {
        size: 224,
    },
}
