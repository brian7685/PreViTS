from .video import Sample
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class davis:

    def __init__(
            self,
            video_dir: str,
            split='train',
            blacklist=None,
    ):

        video_dir = Path(video_dir)
        self._class_name_list = sorted(set(p.name for p in video_dir.glob('*_video/*')))

        self.class_to_index = {name: i for i, name in enumerate(self._class_name_list)}
        self.index_to_class = self._class_name_list

        self._samples = []
        search_dir: Path = video_dir / f'{split}_video'
        blacklisted_count = 0
        idx = 0
        for video_path in sorted(search_dir.glob(f'*/*')):
            if str(video_path.relative_to(video_dir)) in blacklist:
                blacklisted_count += 1
                continue
            
            class_name = video_path.parts[-2]
            davis_c =  str(video_path.parts[-1]).split('.')[0]
            print('davis_c',davis_c)
            video_name = 'soapbox'#'soccerball'#'surf'#'stroller'#'surf'#'swing'#'classic-car'#'snowboard'#'kid-football'#'dogs-jump'#'mbike-trick'#'cat-girl' #'bike-packing' 
            if video_name != str(davis_c):
               continue
            for i in range(2):
                s = Sample(
                    video_path=str(video_path),
                    class_index=self.class_to_index[class_name],
                ) 
                idx+=1
                #print(idx,video_path)
                self._samples.append(s)
        if not self._samples:
            raise Exception(f'No video found in {search_dir}')
        logger.info(
            f'{split} split: {len(self._class_name_list)} classes, {len(self._samples)} videos. {blacklisted_count} videos blacklisted')

    def __getitem__(self, index: int):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)
