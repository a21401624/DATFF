import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

@PIPELINES.register_module()
class LoadPairedImagesFromFile(LoadImageFromFile):
    # Author: Yuxuan Hu
    # Date: 2022/8/8
    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        assert results['img_prefix1'] is not None
        assert results['img_prefix2'] is not None
        filename1 = osp.join(results['img_prefix1'], results['img_info']['filename'])
        filename2 = osp.join(results['img_prefix2'], results['img_info']['filename'])

        img_bytes1 = self.file_client.get(filename1)
        img1 = mmcv.imfrombytes(img_bytes1, flag=self.color_type)
        if self.to_float32:
            img1 = img1.astype(np.float32)

        img_bytes2 = self.file_client.get(filename2)
        img2 = mmcv.imfrombytes(img_bytes2, flag=self.color_type)
        if self.to_float32:
            img2 = img2.astype(np.float32)

        results['filename1'] = filename1
        results['filename2'] = filename2
        results['img1'] = img1
        results['img2'] = img2
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        results['img_fields'] = ['img1', 'img2']
        return results