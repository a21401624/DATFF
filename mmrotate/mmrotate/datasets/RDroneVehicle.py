# Author: Yuxuan Hu
# Modified: 2022/8/18
#           2023/7/15
#           2024/3/26
# Bug Fix: 2022/9/16 
# Test images with no GT bboxes.
import os
import os.path as osp
from multiprocessing import Pool

import torch
import numpy as np
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import poly2obb_np
from mmrotate.core import RBboxOverlaps2D
from mmrotate.core import eval_rbbox_map
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class RDroneVehicleDataset(CustomDataset):
    """Modified from DOTA Dataset.

    Args:
        img_list (str): list of images used, txt file.
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
        class_proj (dict, optional): {'depreciated_cls_name':'new_cls_name'}
    """
    CLASSES = ("car", "truck", "bus", "van", "freight car")

    def __init__(self,
                 img_list,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 class_proj={},
                 **kwargs):
        self.img_list = img_list
        self.version = version
        self.difficulty = difficulty

        self.class_proj = class_proj
        if len(self.class_proj) != 0:
            self.NEW_CLASSES = []
            for i in range(len(self.CLASSES)):
                if self.CLASSES[i] not in self.class_proj.keys():
                    self.NEW_CLASSES.append(self.CLASSES[i])
            self.CLASSES = tuple(self.NEW_CLASSES)

        super(RDroneVehicleDataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Params:
                ann_folder: folder that contains DroneVehicle annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        with open(self.img_list, "r") as f:
            lines = f.readlines()
        ann_files = [ann_folder + line.strip("\n")+'.txt' for line in lines]
        data_infos = []
        if not ann_files:  # test phase
            with open(self.img_list, "r") as f:
                lines = f.readlines()
            for line in lines:
                data_info = {}
                img_id = line.strip("\n")
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        assert len(bbox_info)==10 or len(bbox_info)==11
                        if len(bbox_info)==10:
                            cls_name = bbox_info[8]
                            difficulty = int(bbox_info[9])
                        elif len(bbox_info)==11:
                            cls_name = ' '.join(bbox_info[8:10])
                            difficulty = int(bbox_info[10])

                        if cls_name in self.class_proj.keys():
                            cls_name = self.class_proj[cls_name]

                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)

            # If the results have logvar, we split the bbox results.
            if type(results[0][0]) == tuple:
                results_rbbox = []
                for result in results:
                    result_bbox = []
                    for i in range(len(result)):
                        result_bbox.append(result[i][0])
                    results_rbbox.append(result_bbox)
            else:
                results_rbbox = results

            mean_ap, _ = eval_rbbox_map(
                results_rbbox,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                use_07_metric=False,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

    def PR_for_all_images(self, results, iou_thr=0.5):
        """Calculate the mean Precision and Recall of all classes of all images in the dataset.

        Args:
            results: list[list[array]], output of 'single_gpu_result'.
        """
        PR_result = []
        num_imgs = len(results)
        for i in range(num_imgs):
            PR_for_an_image = self.PR_for_an_image(results[i], self.get_ann_info(i), iou_thr)
            tp_sum = PR_for_an_image['TP']
            fp_sum = PR_for_an_image['FP']
            gt_sum = PR_for_an_image['GTs']
            MisClsFP = PR_for_an_image['MisClsFP']
            MisClsType = PR_for_an_image['MisClsType']
            if (tp_sum + fp_sum)!=0:
                precision = tp_sum / (tp_sum + fp_sum)
            else:
                precision = -1
            if gt_sum!=0:
                recall = tp_sum / gt_sum
            else:
                recall = -1
            filename = self.data_infos[i]['filename']
            PR_result.append({'filename':filename,
                              'dets': int(tp_sum + fp_sum),
                              'TPs': int(tp_sum),
                              'FPs': int(fp_sum),
                              'MisClsFPs': int(MisClsFP), 
                              'GTs': int(gt_sum),
                              'precision': format(precision, '.3f'), 
                              'recall': format(recall, '.3f'),
                              'MisClsType': MisClsType})
        return PR_result

    # def PR_for_an_image(self, result, annotation, iou_thr):
    #     """Count TP and FP detection bboxes of an image. 
    #        For every FP bbox, if it has IoU > 0.8 with an TP bbox, it is misclassified. We count these FPs.

    #     Args:
    #         result: list[array], array dtype float32, the length of result equals the num of classes. 
    #             Every array in the list has shape (num_dets, 6), where num_dets is number of bboxes of this class.
    #             First five cols are bbox coordinates and the last col is the score.
    #         annotations: dict with keys 
    #             'bboxes', 'labels', 'polygons', 'bboxes_ignore', 'labels_ignore', 'polygons_ignore'
    #             annotations[bboxes] is an array with shape (num_gts, 5) representing the GT coordinates.
    #             annotations[labels] is an int64 array with shape (num_gts,) representing the GT class.
    #             Every int64 integer in this array starts from 0 and ends with num_classes-1.
    #     """
    #     num_classes = len(self.CLASSES)
    #     TPs = []
    #     TP_labels = []
    #     FPs = []
    #     FP_labels = []
    #     num_GTs = 0
    #     for i in range(num_classes):
    #         cls_dets, cls_gts, cls_gts_ignore = \
    #             get_cls_results([result], [annotation], i)
    #         tp, fp = tpfp_default(cls_dets[0], cls_gts[0], cls_gts_ignore[0], iou_thr)
    #         num_GTs += len(cls_gts[0])
    #         TPs.append(cls_dets[0][tp[0].astype(np.bool8)])
    #         TP_labels += [i] * int(np.sum(tp))
    #         FPs.append(cls_dets[0][fp[0].astype(np.bool8)])
    #         FP_labels += [i] * int(np.sum(fp))
    #     # concatenate all TPs and FPs, to calculate IoU.
    #     # TPs and FPs shape: (1, #TPs/#FPs, 6)
    #     TPs = np.concatenate(TPs, axis=0)
    #     num_TPs = TPs.shape[0]
    #     FPs = np.concatenate(FPs, axis=0)
    #     num_FPs = FPs.shape[0]
    #     Misclassified_FP = 0
    #     Misclassified_type = []
    #     if num_TPs > 0 and num_FPs > 0:
    #         TPs = torch.from_numpy(TPs).float()
    #         FPs = torch.from_numpy(FPs).float()
        
    #         iou_cal = RBboxOverlaps2D()
    #         iou = iou_cal(TPs, FPs).numpy() # shape:(#TPs, #FPs)
    #         assert iou.shape[0] == num_TPs
    #         assert iou.shape[1] == num_FPs
    #         for i in range(iou.shape[1]):
    #             if np.max(iou[:,i]) > 0.8:
    #                 Misclassified_FP += 1
    #                 Misclassified_type.append(
    #                     str(TP_labels[np.argmax(iou[:,i])]) +
    #                     str(FP_labels[i])
    #                 )
    #     PR_result = {'TP': num_TPs, 
    #                  'FP': num_FPs, 
    #                  'GTs': num_GTs, 
    #                  'MisClsFP': Misclassified_FP,
    #                  'MisClsType': Misclassified_type}
    #     return PR_result

    # def fast_eval_recall(self, proposals, proposal_nums, iou_thrs, logger=None):
    #     """Calculate COCO style AR of the dataset given the proposals.
    #        Modified from mmdet/datasets/coco/fast_eval_recall.
    #        Date: 2022/6/30
    #     """
    #     gt_rbboxes = []
    #     for i in range(len(self)):
    #         ann_info = self.get_ann_info(i)
    #         rbboxes = ann_info['bboxes']
    #         rbboxes = np.array(rbboxes, dtype=np.float32)
    #         if rbboxes.shape[0] == 0:
    #             rbboxes = np.zeros((0, 5))
    #         gt_rbboxes.append(torch.from_numpy(rbboxes))

    #     recalls = eval_recalls(gt_rbboxes, proposals, proposal_nums, iou_thrs, logger=logger)
    #     ar = recalls.mean(axis=1)
    #     return ar