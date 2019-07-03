import torch
import math
import torch.nn as nn


class AssisExc(nn.Module):

    def __init__(self):
        super(AssisExc, self).__init__()

    def forward(self, x, gt_bboxes, stride, epoch):
        num_imgs = x.size(0)
        excitations = torch.zeros_like(x).detach()
        for i in range(num_imgs):
            gt_bbox = gt_bboxes[i] / stride
            inside_gt = torch.zeros_like(x[i]).detach()
            for bbox in gt_bbox:
                bbox[:2] = bbox[:2].floor()
                bbox[2:] = bbox[2:].ceil()
                bbox = bbox.long()
                inside_gt[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            e = (x[i].detach() * inside_gt).mean(dim=0, keepdim=True)
            alpha = 0.5 * (1 + math.cos(math.pi * epoch / 22.))
            excitations[i] = e.expand_as(excitations[i]) * alpha
        print(excitations)
        return x + excitations
