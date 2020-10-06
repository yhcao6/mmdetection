import mmcv

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class VehicleDataset(CustomDataset):

    CLASSES = ('bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
               'truck')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        return mmcv.load(ann_file)['images']
