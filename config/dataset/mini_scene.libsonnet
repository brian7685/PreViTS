local normalization = import "normalization.libsonnet";

{
    name: 'mini_scene',
    root: 'data/mini_scene',
    num_classes: 365,
    annotation_path: 'data/mini_scene/annotations',
    blacklist: [
        'train_video/playing_monopoly/NLL667uPWVA.mp4',
        'train_video/marching/uLaU_15HYdo_000002_000012.mp4'
    ],
    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}