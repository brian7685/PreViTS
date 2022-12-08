local normalization = import "normalization.libsonnet";

{
    name: 'vgg_sound',
    root: 'data/vgg_sound_RSP',
    num_classes: 309,
    blacklist: [
        'train_video/playing_monopoly/NLL667uPWVA.mp4',
        'train_video/marching/uLaU_15HYdo_000002_000012.mp4'
    ],
    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}