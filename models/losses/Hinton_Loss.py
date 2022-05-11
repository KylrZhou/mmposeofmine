import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class HintonLoss(nn.Module):
    def __init__(self,
                 temperature = 5,
                 alpha = 0.5,):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def criterion(self, pred, target, gr, temperature):
        pred_temp_softed = F.softmax(pred / temperature, dim = -1)
        target_temp_softed = F.softmax(target / temperature, dim = -1)

        pred_softed = F.softmax(pred, dim = -1)
        gr_softed = F.softmax(gr, dim = -1)

        soft_loss = F.cross_entropy(pred_temp_softed,target_temp_softed)
        hard_loss = F.cross_entropy(pred_softed,gr_softed)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    #def forward(self, pred, target, temperture):

    def forward(self, output, target, gt, temperture):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_target = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = gt.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmaps_pred = heatmaps_pred[idx].squeeze(1)
            heatmaps_target = heatmaps_target[idx].squeeze(1)
            heatmaps_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmaps_pred,
                                       heatmaps_target,
                                       heatmaps_gt,
                                       temperture)
            else:
                loss += self.criterion(heatmaps_pred,
                                       heatmaps_target,
                                       heatmaps_gt,
                                       temperture)

        return loss / num_joints



