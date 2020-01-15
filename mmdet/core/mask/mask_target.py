import torch
from torch.nn.modules.utils import _pair

from mmdet.ops import roi_align_v2


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets)).float()
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    pos_assigned_gt_inds = pos_assigned_gt_inds
    gt_masks = gt_masks[pos_assigned_gt_inds]
    mask_targets = gt_masks.crop_and_resize(
        pos_proposals, cfg.mask_size).to(device=pos_proposals.device)
    return mask_targets


def ori_mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    fake_inds = (
        torch.arange(num_pos,
                     device=device).to(dtype=pos_proposals.dtype)[:, None])
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    rois = rois.to(device=device)
    if num_pos > 0:
        gt_masks_th = (
            torch.from_numpy(gt_masks).to(device).index_select(
                0, pos_assigned_gt_inds).to(dtype=rois.dtype))
        # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
        targets = (
            roi_align_v2(gt_masks_th[:, None, :, :], rois, mask_size[::-1],
                         1.0, 0, True).squeeze(1))
        # It is important to set the target > threshold rather than >= (~0.5mAP)
        mask_targets = (targets >= 0.5).float()
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
    return mask_targets
