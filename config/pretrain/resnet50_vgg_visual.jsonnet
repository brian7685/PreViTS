local base = import "moco-train-base_vggs_visual.jsonnet";

base {
    batch_size: 8,
    num_workers: 8,

    arch: 'resnet50',
    spatial_transforms+: {
        size: 224,
    },

    
}
