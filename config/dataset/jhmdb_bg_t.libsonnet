local normalization = import "normalization.libsonnet";

{
    name: 'jhmdb',
    root: 'data/jhmdb_bg_t/videos',
    annotation_path: 'data/jhmdb_fg/metafile',
    fold: 1,
    num_classes: 21,

    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}