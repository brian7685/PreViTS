local normalization = import "normalization.libsonnet";

{
    name: 'vgg_sound_debug',
    root: 'data/vgg_sound_debug',
    num_classes: 309,
    blacklist: [
        'train_video/playing_monopoly/NLL667uPWVA.mp4',
        'train_video/frog croaking/twOKVLUzZLQ_000000.mp4'
    ],
    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}