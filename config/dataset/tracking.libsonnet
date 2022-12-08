local normalization = import "normalization.libsonnet";

{
    name: 'trackingNet',
    root: 'data/trackingNet',
    num_classes: 1,
    blacklist: [],
    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}