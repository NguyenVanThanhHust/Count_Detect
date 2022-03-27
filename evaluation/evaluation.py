# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
from os.path import join

from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.evaluator import DatasetEvaluator
import os.path as osp
import cv2
import argparse
import pickle as pkl

class COCOEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(self, gt_json_file, pred_json_file, img_folder=None, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = ["bbox", ]
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        # replace fewx with d2
        self._logger = logging.getLogger(__name__)

        gt_json_file = PathManager.get_local_path(gt_json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(gt_json_file)

        pred_json_file = PathManager.get_local_path(pred_json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self.pred_coco_api = COCO(pred_json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        self.counting_dict = dict()
        self._predictions = []
        self.img_folder = img_folder
        self._vis_dir = osp.join(self._output_dir, "vis_res")
        os.makedirs(self._vis_dir, exist_ok=True)
        self.aps = []
        self.im_ids = self.pred_coco_api.getImgIds()
        self._image_set = None

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        return tasks

    def process(self, ):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        print("number of images", len(self.im_ids))
        for img_id in self.im_ids:
            anno_ids = self.pred_coco_api.getAnnIds([img_id])
            pred_annos = self.pred_coco_api.loadAnns(anno_ids)
            prediction = {"image_id": img_id}
            results = []
            num_pred = len(pred_annos)

            for anno in pred_annos:
                box = anno["bbox"]
                x1, y1, x2, y2 = box
                xmin, ymin, = x1, y1
                w, h = x2 - x1, y2 - y1
                new_box = [int(xmin), int(ymin), int(w), int(h)]

                result = {
                    "image_id": anno["image_id"],
                    "category_id": anno["category_id"],
                    "bbox": new_box,
                    "score": anno["score"],
                }
                results.append(result)

            gt_anno_ids = self._coco_api.getAnnIds([img_id])
            gt_annos = self._coco_api.loadAnns(gt_anno_ids)

            if self.img_folder is not None:
                gt_info = self._coco_api.loadImgs([img_id])
                img_name = gt_info[0]["file_name"]
                img_path = join(self.img_folder, img_name)
                img = cv2.imread(img_path)
                for anno in pred_annos:
                    box = anno["bbox"]
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                for anno in gt_annos:
                    box = anno["bbox"]
                    x1, y1, w, h = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x1+w), int(y1+h)
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                output_img_path = join(self._vis_dir, img_name)
                cv2.imwrite(output_img_path, img)
                
            prediction["instances"] = results
            self._predictions.append(prediction)
            self.counting_dict[img_id] = {"gt": len(gt_annos), "pred": num_pred} 

    def evaluate(self):
        predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        self._eval_predictions(set(self._tasks), predictions)

        # Copy so the caller can do whatever with results
        cnt = 0
        SAE = 0  # sum of absolute errors
        SSE = 0  # sum of square errors
        for img_id, anno in self.counting_dict.items():
            gt_cnt = anno["gt"]
            pred_cnt = anno["pred"]
            cnt = cnt + 1
            err = abs(gt_cnt - pred_cnt)
            SAE += err
            SSE += err**2
        print(cnt, SAE/cnt, (SSE/cnt)**0.5)
        print(self._results)

        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return
        
        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            if self._image_set is not None:
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, coco_results, task, self._image_set 
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
            else:
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, coco_results, task, 
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
            
            res = self._derive_coco_results(
                # coco_eval, task, class_names=self._metadata.get("thing_classes")
                coco_eval, task, class_names=["fg", ]
            )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")
        if class_names is None or len(class_names) <= 1:
            return results

        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa

        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


class COCOevalMaxDets(COCOeval):
    """
    Modified version of COCOeval for evaluating AP with a custom
    maxDets (by default for COCO, maxDets is 100)
    """

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results given
        a custom value for  max_dets_per_image
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=1000):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            # Evaluate AP using the custom limit on maximum detections per image
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, img_ids=None, max_dets_per_image=None, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0
    coco_dt = coco_gt.loadRes(coco_results)
    # coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)
    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    # For COCO, the default max_dets_per_image is [1, 10, 100].
    if max_dets_per_image is None:
        max_dets_per_image = [900, 1000, 1100]  # from datasets
    else:
        assert (
            len(max_dets_per_image) >= 3
        ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
        # In the case that user supplies a custom input for max_dets_per_image,
        # apply COCOevalMaxDets to evaluate AP with the custom input.
        if max_dets_per_image[2] != 100:
            coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)
    if iou_type != "keypoints":
        coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval

def get_args_parser():
    parser = argparse.ArgumentParser('Count and Detect', add_help=False)
    parser.add_argument('--gt_json_path', default="./gts.json", type=str)
    parser.add_argument('-p', '--pred_json_path', default="./outputs/predictions.json", type=str)
    parser.add_argument('-o', '--output_dir', default="./outputs/visualize/", type=str)
    parser.add_argument('--img_folder', default=None, type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args_parser()
    gt_json_path = args.gt_json_path
    pred_json_path = args.pred_json_path
    output_dir = args.output_dir
    img_folder = args.img_folder
    coco_evaluator = COCOEvaluator(gt_json_file = gt_json_path, 
                            pred_json_file = pred_json_path, 
                            output_dir=output_dir,  
                            img_folder=img_folder)
    coco_evaluator.process()
    coco_evaluator.evaluate()
