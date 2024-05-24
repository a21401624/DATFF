# Date: 2023/2/21
import torch
import torch.nn as nn

from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from .utils import FeatureRefineModule
from mmcv.runner import auto_fp16


@ROTATED_DETECTORS.register_module()
class BimodalR3Det(RotatedBaseDetector):
    """Rotated Refinement RetinaNet."""

    def __init__(self,
                 num_refine_stages,
                 backbone1,
                 backbone2,
                 fusion_neck=None,
                 neck=None,
                 bbox_head=None,
                 frm_cfgs=None,
                 refine_heads=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BimodalR3Det, self).__init__()

        backbone1.pretrained = pretrained
        self.backbone1 = build_backbone(backbone1)
        backbone2.pretrained = pretrained
        self.backbone2 = build_backbone(backbone2)
        self.num_refine_stages = num_refine_stages
        if fusion_neck is not None:
            self.fusion_neck = build_neck(fusion_neck)
        if neck is not None:
            self.neck = build_neck(neck)
        if train_cfg is not None:
            bbox_head.update(train_cfg=train_cfg['s0'])
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.init_weights()
        self.feat_refine_module = nn.ModuleList()
        self.refine_head = nn.ModuleList()
        for i, (frm_cfg,
                refine_head) in enumerate(zip(frm_cfgs, refine_heads)):
            self.feat_refine_module.append(FeatureRefineModule(**frm_cfg))
            if train_cfg is not None:
                refine_head.update(train_cfg=train_cfg['sr'][i])
            refine_head.update(test_cfg=test_cfg)
            self.refine_head.append(build_head(refine_head))
        for i in range(self.num_refine_stages):
            self.feat_refine_module[i].init_weights()
            self.refine_head[i].init_weights()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat_train(self, img1, img2, **kwargs):
        """Directly extract features from the backbone+neck."""
        x1 = self.backbone1(img1)
        x2 = self.backbone2(img2)
        if self.fusion_neck.has_train_loss():
            fusion_loss, x = self.fusion_neck.forward_train(x1, x2, **kwargs)
        else:
            x = self.fusion_neck(x1, x2, **kwargs)
            fusion_loss = {}
        if self.with_neck:
            x = self.neck(x)
        return fusion_loss, x

    def extract_feat(self, img1, img2):
        """Directly extract features from the backbone+neck."""
        x1 = self.backbone1(img1)
        x2 = self.backbone2(img2)
        if self.fusion_neck.has_train_loss():
            x = self.fusion_neck.forward_test(x1, x2)
        else:
            x = self.fusion_neck(x1, x2)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)
        return outs

    def forward_train(self,
                      img1,
                      img2,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward function."""
        losses = dict()
        fusion_loss, x = self.extract_feat_train(img1, img2)
        losses.update(fusion_loss)

        outs = self.bbox_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses[f's0.{name}'] = value

        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            lw = self.train_cfg.stage_loss_weights[i]

            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss_refine = self.refine_head[i].loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
            for name, value in loss_refine.items():
                losses[f'sr{i}.{name}'] = ([v * lw for v in value]
                                           if 'loss' in name else value)

            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        return losses

    def simple_test(self, img1, img2, img_meta, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img1, img2)
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.refine_head[-1].get_bboxes(*bbox_inputs, rois=rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels,
                         self.refine_head[-1].num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def forward_test(self, imgs1, imgs2, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs1, 'imgs1'), (imgs2, 'imgs2'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs1)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs1)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img1, img_meta in zip(imgs1, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img1.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            if 'entropy_map1' in kwargs:
                kwargs['entropy_map1'] = kwargs['entropy_map1'][0]
            if 'entropy_map2' in kwargs:
                kwargs['entropy_map2'] = kwargs['entropy_map2'][0]
            return self.simple_test(imgs1[0], imgs2[0], img_metas[0], **kwargs)
        else:
            assert imgs1[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs1[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs1, imgs2, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img1, img2, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img1[0], img2[0], img_metas[0])

        if return_loss:
            return self.forward_train(img1, img2, img_metas, **kwargs)
        else:
            return self.forward_test(img1, img2, img_metas, **kwargs)
