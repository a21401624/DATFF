# Author: Hu Yuxuan
# Date: 2022/3/1
# Modified: 2022/8/18
import warnings

import mmcv

from .builder import DATASETS
from mmdet.datasets.pipelines import Compose
from .DroneVehicle import DroneVehicleDataset


@DATASETS.register_module()
class BimodalDroneVehicleDataset(DroneVehicleDataset):
    """Load a pair of images, with one set of annotations. 
        
    Args:
    """

    def __init__(self,
                 img_list,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 class_proj={},
                 classes=None,
                 img_prefix1='',
                 img_prefix2='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.img_list = img_list
        self.ann_file = ann_file
        self.version = version
        self.difficulty = difficulty
        self.img_prefix1 = img_prefix1
        self.img_prefix2 = img_prefix2
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)

        self.class_proj = class_proj
        if len(self.class_proj) != 0:
            self.NEW_CLASSES = []
            for i in range(len(self.CLASSES)):
                if self.CLASSES[i] not in self.class_proj.keys():
                    self.NEW_CLASSES.append(self.CLASSES[i])
            self.CLASSES = tuple(self.NEW_CLASSES)

        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix1'] = self.img_prefix1
        results['img_prefix2'] = self.img_prefix2
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []