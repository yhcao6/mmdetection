import torch

from mmdet.core import bbox_overlaps, delta2bbox


def isr_p(cls_score,
          bbox_pred,
          bbox_targets,
          rois,
          sampling_results,
          loss_cls,
          target_means=(0, 0, 0, 0),
          target_stds=(0.1, 0.1, 0.2, 0.2),
          k=2,
          bias=0):
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets
    pos_label_inds = ((labels >= 0) & (labels < 80)).nonzero().reshape(-1)
    pos_labels = labels[pos_label_inds]
    num_pos = float(pos_label_inds.size(0))
    if num_pos == 0:
        return label_weights

    gts = list()
    last_max_gt = 0
    for i in range(len(sampling_results)):
        gt_i = sampling_results[i].pos_assigned_gt_inds + last_max_gt
        gts.append(gt_i)
        last_max_gt = gt_i.max() + 1
    gts = torch.cat(gts)
    assert len(gts) == num_pos

    # ori loss cls
    cls_score = cls_score.detach()
    bbox_pred = bbox_pred.detach()

    # calculate positive box ious
    if rois.size(-1) == 5:
        pos_rois = rois[pos_label_inds][:, 1:]
    else:
        pos_rois = rois[pos_label_inds]
    if bbox_pred.size(-1) > 4:
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        pos_delta_pred = bbox_pred[pos_label_inds, pos_labels].view(-1, 4)
    else:
        pos_delta_pred = bbox_pred[pos_label_inds].view(-1, 4)

    pos_delta_target = bbox_targets[pos_label_inds].view(-1, 4)
    pos_bbox_pred = delta2bbox(pos_rois, pos_delta_pred, target_means,
                               target_stds)
    target_bbox_pred = delta2bbox(pos_rois, pos_delta_target, target_means,
                                  target_stds)
    ious = bbox_overlaps(pos_bbox_pred, target_bbox_pred, is_aligned=True)

    pos_label_weights = label_weights[pos_label_inds]
    # rank as class gt iou
    max_l_num = pos_labels.bincount().max()
    for l in pos_labels.unique():
        l_inds = (pos_labels == l).nonzero().view(-1)
        l_gts = gts[l_inds]
        for t in l_gts.unique():
            t_inds = l_inds[l_gts == t]
            t_ious = ious[t_inds]
            _, t_iou_rank_idx = t_ious.sort(descending=True)
            _, t_iou_rank = t_iou_rank_idx.sort()
            ious[t_inds] += max_l_num - t_iou_rank.float()
        l_ious = ious[l_inds]
        _, l_iou_rank_idx = l_ious.sort(descending=True)
        _, l_iou_rank = l_iou_rank_idx.sort()
        pos_label_weights[l_inds] *= (max_l_num -
                                      l_iou_rank.float()) / max_l_num

    pos_label_weights = (bias + pos_label_weights * (1 - bias)).pow(k)

    pos_loss_cls = loss_cls(
        cls_score[pos_label_inds], pos_labels, reduction_override='none')
    ori_pos_loss_cls = pos_loss_cls * label_weights[pos_label_inds]
    new_pos_loss_cls = pos_loss_cls * pos_label_weights
    pos_loss_cls_ratio = ori_pos_loss_cls.sum() / new_pos_loss_cls.sum()
    pos_label_weights *= pos_loss_cls_ratio
    label_weights[pos_label_inds] = pos_label_weights

    debug = False
    if debug:
        p = delta2bbox(
            pos_rois, pos_delta_pred, means=target_means, stds=target_stds)
        t = delta2bbox(
            pos_rois, pos_delta_target, means=target_means, stds=target_stds)
        ious = bbox_overlaps(p, t, is_aligned=True)

        pos_labels = labels[pos_label_inds]
        for l in pos_labels.unique():
            l_inds = (pos_labels == l).nonzero().view(-1)
            l_gt = gts[l_inds]
            for t in l_gt.unique():
                t_inds = l_inds[l_gt == t]
                _, t_iou_rank_idx = ious[t_inds].sort(descending=True)
                print('label: {}, gt: {}'.format(l, t))
                print('ious: {}'.format(ious[t_inds[t_iou_rank_idx]]))
                print('weight: {}'.format(
                    label_weights[pos_label_inds][t_inds][t_iou_rank_idx]))
        print('neg weight: {}'.format(label_weights[labels == 80][0:5]))
        print('num pos: {}, num sample pos: {}'.format(
            (labels < 80).sum(), (label_weights[labels < 80] > 0).sum()))
        print('num neg: {}, num sample neg: {}\n'.format(
            (labels == 80).sum(), (label_weights[labels == 80] > 0).sum()))
        exit()

    bbox_targets = labels, label_weights, bbox_targets, bbox_weights
    info = dict(k1=torch.Tensor([k]).cuda(), b1=torch.Tensor([bias]).cuda())
    return bbox_targets, info


def carl_loss(cls_score,
              labels,
              bbox_pred,
              bbox_targets,
              loss_bbox,
              k=1,
              bias=0.2,
              detach=False,
              normalize=False,
              avg_factor=None,
              inplace=False):

    pos_label_inds = ((labels >= 0) & (labels < 80)).nonzero().reshape(-1)
    pos_labels = labels[pos_label_inds]

    # multiply pos cls score into corresponding bbox weight and remain gradient
    pos_cls_score = cls_score.softmax(-1)[pos_label_inds, pos_labels]
    pos_bbox_weights = (bias + (1 - bias) * pos_cls_score).pow(k)

    # weight ratio normalize
    num_pos = float(pos_cls_score.size(0))
    weight_ratio = num_pos / pos_bbox_weights.sum()
    pos_bbox_weights *= weight_ratio

    # loss ratio normalize
    if detach:
        bbox_pred = bbox_pred.detach()
    if avg_factor is None:
        avg_factor = bbox_targets.size(0)
    if bbox_pred.size(-1) > 4:
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        ori_loss_reg = loss_bbox(
            bbox_pred[pos_label_inds, pos_labels],
            bbox_targets[pos_label_inds],
            reduction_override='none') / avg_factor
    else:
        ori_loss_reg = loss_bbox(
            bbox_pred[pos_label_inds],
            bbox_targets[pos_label_inds],
            reduction_override='none') / avg_factor
    score_weights = pos_bbox_weights[:, None].expand(-1, 4)
    if inplace:
        new_loss = ori_loss_reg * score_weights
        norm_ratio = ori_loss_reg.sum() / new_loss.sum()
        carl_loss = new_loss.sum() * norm_ratio
    else:
        carl_loss = (ori_loss_reg * score_weights).sum()

    debug = False
    if debug:
        pos_cls_score = cls_score.softmax(-1)[pos_label_inds, pos_labels]
        _, score_rank_idx = pos_cls_score.sort(descending=True)
        print('score: {}'.format(pos_cls_score[score_rank_idx]))
        print('weight: {}'.format(pos_bbox_weights[score_rank_idx]))
        exit()
    return dict(loss_carl=carl_loss[None]), dict(
        k2=torch.Tensor([k]).cuda(), b2=torch.Tensor([bias]).cuda())
