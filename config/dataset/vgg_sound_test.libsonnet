local normalization = import "normalization.libsonnet";

{
    name: 'vgg_sound_test',
    root: 'data/vgg_sound_RSP_op',
    num_classes: 309,
    blacklist: [
        'train_video/playing_monopoly/NLL667uPWVA.mp4',
        'train_video/marching/uLaU_15HYdo_000002_000012.mp4'
    ],
    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}