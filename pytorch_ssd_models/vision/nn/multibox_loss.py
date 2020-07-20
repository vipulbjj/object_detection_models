import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from ..utils import box_utils
# from focalloss import *


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
#         print(confidence.shape, labels.shape, confidence, labels)32,3000,3 32,3000
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
#             loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
#             print(F.log_softmax(confidence, dim=2).shape)(32,3000,3)
#             print(confidence,"initial")
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        ###########################
#             print(mask.shape)(32,3000)
######################################
#             mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
##################################debugging
#         print(mask)boolean array
#####################################
        confidence = confidence[mask, :]
#######################
#         print(confidence.shape)#(1016,3)
##########################
#         print((labels[mask]==0).sum(),"background labels")
#         print((labels[mask]==1).sum(),"left foot")
#         print((labels[mask]==2).sum(),"right foot")
# tensor(1074, device='cuda:0') background labels
# tensor(226, device='cuda:0') left foot
# tensor(132, device='cuda:0') right foot
#         the ratio is actually just 1/6 in my case roughly
#         classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
#         classification_loss = FocalLoss(gamma=0,confidence.reshape(-1, num_classes), labels[mask])
        input_conf = confidence.reshape(-1, num_classes)#nothing happens here
#####################
#         print(confidence.shape)
########################
        
        target = labels[mask]
#         print(target.shape)
#         target = target.view(-1,1)#unknown no. of rows = labels and 1 column
####################
#         print(confidence.dim())2
####################
#         print(input_conf)
#changed weight of bg from 0.01 to 0.2
        weights = torch.Tensor([0.20,1.0,1.0])
        weights = weights.cuda()
        classification_loss=F.cross_entropy(input_conf, target, weight = weights, size_average=False)
#         classification_loss_1=F.cross_entropy(input_conf[:,1], target, size_average=False)
#         classification_loss_2=F.cross_entropy(input_conf[:,2], target, size_average=False)
#         classification_loss=0.001*classification_loss_0+classification_loss_1+classification_loss_2


        if False:#previous implementation of focal loss. won't run now
            logpt = F.log_softmax(input_conf)
    ##################
    #         print(logpt.shape)#(1436,3)
    #         print(labels.shape)#(1436)
    ##################
            logpt=logpt.gather(1,target)#error that index and input should hv same dim
            logpt=logpt.view(-1)
            pt = Variable(logpt.data.exp())
            alpha = 0.3#[0,1]
            gamma = 2#[0,5]
    #         print(pt.shape, logpt.shape)torch.Size([1148])
    #         if alpha is not None:

            classification_loss = -1 * (1-pt)**gamma*logpt
            classification_loss = classification_loss.sum()
#         print(classification_loss.shape)torch.Size([1296])
        
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos
    
    
