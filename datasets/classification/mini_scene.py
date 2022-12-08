from collections import OrderedDict
from pathlib import Path
from typing import Dict

from pyhocon import ConfigTree

from .video import Sample


def get_real_path(path: Path, extensions):
    for ext in extensions:
        testing_path = path.with_suffix(ext)
        if testing_path.exists():
            return testing_path
    raise Exception(f'{path} with possible extensions {extensions} not found')


class mini_scene:
    '''UCF101 samples

    Sample 1 clip per video
    '''

    def __init__(
            self,
            video_dir,
            annotation_dir,
            fold=1,
            split='train',
            extensions=('.avi', '.mp4'),
    ):
        self.annotation_dir = Path(annotation_dir)

        
        list_name = {
            'train': 'train',
            'val': 'test',
            'test': 'test',
        }[split]
        video_list_path = self.annotation_dir / f'{split}.txt'
        # self._samples = []
        samples = []
        with video_list_path.open('r') as f:
            for line in f:
                # Not using class index in list file, because test split don't have it.
                video = (' ').join(line.strip().split(' ')[:-1])
                video_path = video_dir+'/videos/'+split+'/'+video
                #Path(video_dir) / video
                #class_name = video_path.parts[-2]
                class_index = line.strip().split(' ')[-1]#self.class_to_index[class_name]

                s = Sample(
                    video_path=str(video_path),
                    class_index=class_index,
                )
                samples.append(s)

        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index: int):
        return self._samples[index]

    @staticmethod
    def read_class_idx(annotation_dir: Path) -> Dict[int, str]:
        class_ind_path = annotation_dir / 'classInd.txt'
        class_dict = OrderedDict()
        with class_ind_path.open('r') as f:
            for line in f:
                class_idx, class_name = line.strip().split(' ')
                class_dict[int(class_idx)] = class_name
        return class_dict
