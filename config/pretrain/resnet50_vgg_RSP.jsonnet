local base = import "moco-train-base_vggs_RSP.jsonnet";

base {
    batch_size: 16,
    num_workers: 4,

    arch: 'resnet50',
    spatial_transforms+: {
        size: 224,
    },
}
