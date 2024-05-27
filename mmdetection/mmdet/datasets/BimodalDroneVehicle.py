# Author: Yuxuan Hu
# Date: 2024/5/12
from .builder import DATASETS
from .pipelines import Compose
from .DroneVehicle import DroneVehicleDataset


@DATASETS.register_module()
class BimodalDroneVehicleDataset(DroneVehicleDataset):
    """Load a pair of images, with one set of annotation. 
        
    Args:
    """
    
    def __init__(self,
                 img_list,
                 ann_file,
                 pipeline,
                 difficulty=100,
                 classes=None,
                 img_prefix1='',
                 img_prefix2='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.img_list = img_list
        self.ann_file = ann_file
        self.difficulty = difficulty
        self.img_prefix1 = img_prefix1
        self.img_prefix2 = img_prefix2
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not self.test_mode:
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