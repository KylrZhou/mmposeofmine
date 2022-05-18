# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose
import torch

from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


class SE_Block(torch.nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(c, c // r, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(c // r, c, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

@POSENETS.register_module()
class TopDownSFTNSEST(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 teacher_backbone=None,
                 neck=None,
                 keypoint_head=None,
                 teacher_keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 pretrained_teacher_backbone=None,
                 pretrained_teacher_head=None,
                 pretrained_SE_block=None,
                 loss_pose=None,
                 use_se_blocks=False,
                 Temp=10):
        super().__init__()
        self.fp16_enabled = False
        self.use_se_blocks = use_se_blocks
        self.Temp = Temp

        self.backbone = builder.build_backbone(backbone)
        self.teacher_backbone = builder.build_backbone(teacher_backbone)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)
            self.teacher_keypoint_head = builder.build_head(teacher_keypoint_head)
        self.init_weights(pretrained=pretrained)
        self.init_weights_t(backbone=pretrained_teacher_backbone, head=pretrained_teacher_head)
        if self.use_se_blocks == True:
            ckpt_s = torch.load(pretrained_SE_block)
            self.SE1 = SE_Block(c=256)
            self.SE2 = SE_Block(c=512)
            self.SE3 = SE_Block(c=1024)
            self.SE1.load_state_dict(ckpt_s['A'])
            self.SE2.load_state_dict(ckpt_s['B'])
            self.SE3.load_state_dict(ckpt_s['C'])
            self.SE1.eval()
            self.SE2.eval()
            self.SE3.eval()

    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights_teacher()
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    def init_weights_t(self, backbone, head):
        ckpt_b = torch.load(backbone)
        ckpt_h = torch.load(head)
        load_state_dict(self.teacher_backbone, ckpt_b)
        load_state_dict(self.teacher_keypoint_head, ckpt_h)
        self.teacher_backbone.eval()
        self.teacher_keypoint_head.eval()

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths \
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        origin_output = list()
        origin_teacher_output = list()
        tmp = self.backbone(img)
        for i in tmp:
            origin_output.append(i)
        output = self.keypoint_head(origin_output[3])
        with torch.no_grad():
            tmp = self.teacher_backbone(img)
            for i in tmp:
                origin_teacher_output.append(i)
            teacher_output = self.teacher_keypoint_head(origin_teacher_output[3])
            target_t = teacher_output
            if self.use_se_blocks == True:
                origin_teacher_output[0] = self.SE1(origin_teacher_output[0])
                origin_teacher_output[1] = self.SE2(origin_teacher_output[1])
                origin_teacher_output[2] = self.SE3(origin_teacher_output[2])

        # if return loss
        losses = dict()
        if self.with_keypoint:
            origin_teacher_output[0] = torch.nn.functional.softmax(origin_teacher_output[0]/self.Temp, dim=0)
            origin_teacher_output[1] = torch.nn.functional.softmax(origin_teacher_output[1]/self.Temp, dim=0)
            origin_teacher_output[2] = torch.nn.functional.softmax(origin_teacher_output[2]/self.Temp, dim=0)
            origin_teacher_output[3] = torch.nn.functional.softmax(origin_teacher_output[3]/self.Temp, dim=0)
            teacher_output = torch.nn.functional.softmax(teacher_output/self.Temp, dim=0)
            origin_output[0] = torch.nn.functional.log_softmax(origin_output[0]/self.Temp, dim=0)
            origin_output[1] = torch.nn.functional.log_softmax(origin_output[1]/self.Temp, dim=0)
            origin_output[2] = torch.nn.functional.log_softmax(origin_output[2]/self.Temp, dim=0)
            origin_output[3] = torch.nn.functional.log_softmax(origin_output[3]/self.Temp, dim=0)
            student_output = torch.nn.functional.log_softmax(output/self.Temp, dim=0)
            keypoint_losses = dict()
            keypoint_accuracy = dict()
            keypoint_losses['Stage1_loss'] = torch.nn.functional.kl_div(origin_output[0], origin_teacher_output[0], reduction = 'batchmean') * 0.00001
            keypoint_losses['Stage2_loss'] = torch.nn.functional.kl_div(origin_output[1], origin_teacher_output[1], reduction = 'batchmean') * 0.00001
            keypoint_losses['Stage3_loss'] = torch.nn.functional.kl_div(origin_output[2], origin_teacher_output[2], reduction = 'batchmean') * 0.00001
            keypoint_losses['Stage4_loss'] = torch.nn.functional.kl_div(origin_output[3], origin_teacher_output[3], reduction = 'batchmean') * 0.0001
            keypoint_losses['st2th_loss'] = torch.nn.functional.kl_div(student_output, teacher_output, reduction = 'batchmean') * 0.01
            keypoint_losses['GT_loss'] = self.keypoint_head.get_loss(output, target, target_weight) * 0.5
            losses.update(keypoint_losses)
            keypoint_accuracy['st2th_acc'] = self.keypoint_head.get_accuracy(output, target_t, target_weight)
            keypoint_accuracy['acc_pose'] = self.keypoint_head.get_accuracy(output, target, target_weight)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features_tuple = self.backbone(img)
        features = features_tuple[3]
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped_tuple = self.backbone(img_flipped)
            features_flipped = features_flipped_tuple[3]
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap +
                                  output_flipped_heatmap) * 0.5

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
