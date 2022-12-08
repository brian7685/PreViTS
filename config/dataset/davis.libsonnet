local normalization = import "normalization.libsonnet";

{
    name: 'davis',
    root: 'data/davis2016',
    num_classes: 1,
    blacklist: [
    ],
    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}