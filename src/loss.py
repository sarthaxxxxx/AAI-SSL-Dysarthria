import torch
import torch.nn as nn

class AAILoss:
    def __init__(self):
        self.loss = nn.MSELoss(reduction = 'none')

    def compute_loss(
        self, targets, pred
    ):
        loss_dict = {}

        assert targets.shape[-1] == pred.shape[-1], "Targets and predictions must have the same number of features" 

        loss = self.loss(targets, pred)
        loss = torch.sum(torch.mean(loss, dim = 0)) / targets.shape[-1]

        loss_dict['aai'] = loss
        loss_dict['total'] = loss
        return loss_dict