import os.path as osp
import mmcv
import numpy as np
from ..builder import ROTATED_PIPELINES

@ROTATED_PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 backend='cv2',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.backend = backend
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order, backend=self.backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

@ROTATED_PIPELINES.register_module()
class LoadPairedImagesFromFile(LoadImageFromFile):
    # Author: Hu Yuxuan
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