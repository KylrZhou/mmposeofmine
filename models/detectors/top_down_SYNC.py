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

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16

from collections import OrderedDict
from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict
import torch

@POSENETS.register_module()
class TopDownSYNC(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Pat h to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone_front,
                 backbone_back,
                 backbone_teacher,
                 neck=None,
                 keypoint_head=None,
                 keypoint_head_teacher=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None,
                 Temp=10):
        super().__init__()
        self.train_teacher=1
        self.fp16_enabled = False
        self.Temp = Temp
        self.backbone_front = builder.build_backbone(backbone_front)
        self.backbone_back = builder.build_backbone(backbone_back)
        self.backbone_teacher = builder.build_backbone(backbone_teacher)

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
            self.keypoint_head_teacher = builder.build_head(keypoint_head_teacher)


        self.init_weights(pretrained=pretrained)

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
        self.backbone_teacher.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights_teacher()
        if self.with_keypoint:
            self.keypoint_head_teacher.init_weights()

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
            losses=dict()
            tmp = self.forward_train(img, target, target_weight, img_metas, **kwargs)
            losses.update(tmp)
            self.forward_train1(img, target, target_weight, img_metas, **kwargs)
            losses.update(tmp)
            return losses
        return self.forward_test(img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        # if return loss
        losses = dict()
        if self.with_keypoint:
            for i in self.backbone_teacher.parameters():
                i.requires_grad = True
            for i in self.keypoint_head_teacher.parameters():
                i.requires_grad = True
            keypoint_losses = dict()
            keypoint_accuracy = dict()
            for i in self.backbone_front.parameters():
                i.requires_grad = False
            for i in self.backbone_back.parameters():
                i.requires_grad = False
            for i in self.keypoint_head_teacher.parameters():
                i.requires_grad = False
            origin_teacher_output = self.backbone_teacher(img)
            teacher_output = self.keypoint_head_teacher(origin_teacher_output[3])
            keypoint_losses['T_gt_loss'] = self.keypoint_head.get_loss(teacher_output, target, target_weight)
            for i in self.backbone_front.parameters():
                i.requires_grad = True
            for i in self.backbone_back.parameters():
                i.requires_grad = True
            for i in self.keypoint_head_teacher.parameters():
                i.requires_grad = True
            student_output = self.backbone_back(origin_teacher_output[1])
            student_output = self.keypoint_head(student_output[1])
            teacher_soft = torch.nn.functional.softmax(teacher_output/self.Temp, dim=0)
            student_soft = torch.nn.functional.log_softmax(student_output/self.Temp, dim=0)
            keypoint_losses['T_kl_loss'] = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction = 'batchmean')
            losses.update(keypoint_losses)
            keypoint_accuracy['T_loss'] = keypoint_losses['T_gt_loss'] + keypoint_losses['T_kl_loss']
            keypoint_accuracy['T_t_accuracy'] = self.keypoint_head.get_accuracy(teacher_output, target, target_weight)
            keypoint_accuracy['T_s_accuracy'] = self.keypoint_head.get_accuracy(student_output, target, target_weight)
            losses.update(keypoint_accuracy)
        return losses

    def forward_train1(self, img, target, target_weight, img_metas, **kwargs):
        losses=dict()
        if self.with_keypoint:
            keypoint_losses = dict()
            keypoint_accuracy = dict()
            for i in self.backbone_teacher.parameters():
                i.requires_grad = False
            for i in self.keypoint_head_teacher.parameters():
                i.requires_grad = False
            origin_student_output = self.backbone_front(img)
            origin_student_output = self.backbone_back(origin_student_output[1])
            student_output = self.keypoint_head(origin_student_output[1])
            keypoint_losses['S_gt_loss'] = self.keypoint_head.get_loss(student_output, target, target_weight)
            teacher_soft = torch.nn.functional.softmax(origin_teacher_output[2]/self.Temp, dim=0)
            student_soft = torch.nn.functional.log_softmax(origin_student_output[0]/self.Temp, dim=0)
            keypoint_losses['S_ft_loss'] = torch.nn.functional.kl_div(student_soft, teacher_soft, reduction = 'batchmean')
            keypoint_losses['S_hm_loss'] = self.keypoint_head.get_loss(student_output, teacher_output, target_weight)
            losses.update(keypoint_losses)
            keypoint_accuracy['S_loss'] = keypoint_losses['S_gt_loss'] + keypoint_losses['S_ft_loss'] + keypoint_losses['S_hm_loss']
            keypoint_accuracy['S_t_accuracy'] = self.keypoint_head.get_accuracy(teacher_output, target, target_weight)
            keypoint_accuracy['S_s_accuracy'] = self.keypoint_head.get_accuracy(student_output, target, target_weight)
            losses.update(keypoint_accuracy)
        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone_front(img)
        features = self.backbone_back(features[1])
        features = features[1]
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)
            features_flipped = self.backbone_front(img_flipped)
            features_flipped = self.backbone_back(features_flipped[1])
            features_flipped = features_flipped[1]
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
