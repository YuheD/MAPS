import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
# import logging
import os

# logger = logging.getLogger(__name__)

def _init_weight(modules):
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def num_trainable_params(model, print_list = False):
    """Count number of trainable parameters
    """
    n_trainable = 0
    n_total = 0
    #for child in model.children():
    for param in model.parameters():
        n_total += param.nelement()
        if param.requires_grad == True:
            n_trainable += param.nelement()
    print('Trainable {:,} parameters out of {:,}'.format(n_trainable, n_total))
    if print_list:
        print('Trainable parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('\t {} \t {} \t {:,}'.format(name, param.size(), param.numel()))
    return n_trainable

class KpClassNet(nn.Module):
    """Classify keypoint representation
    """
    def __init__(self, in_feat, out_feat, out_class, inter_feat=0, tune_hm=False, conv1x1=True):
        super(KpClassNet, self).__init__()
        self.in_feat = in_feat
        self.conv1x1 = conv1x1

        if self.conv1x1:
            self.conv1x1_layer = nn.Conv2d(in_feat, out_feat, 1, stride=1, padding=0)

        self.gmp = nn.AdaptiveMaxPool2d(1)
        #self.bn_bottleneck = nn.BatchNorm1d(in_feat)
        if inter_feat > 0:
            self.classifier = nn.Sequential(
                                nn.Linear(out_feat, inter_feat, bias=True),
                                nn.ReLU(),
                                nn.Linear(inter_feat, out_class, bias=True)
                                )
        else:
            self.classifier = nn.Linear(out_feat, out_class, bias=True)

        # logger.info('Initialised keypoint classification model with {} params'.format(num_trainable_params(self.classifier)))

    def forward(self, feat, hm):
        """
        feat: tensor of shape (bs, num_feat, h, w)
        hm: tensor of shape (bs, 1, h, w)
        """
        assert feat.shape[0] == hm.shape[0]
        assert feat.shape[1] == self.in_feat, 'Expected {} features in dim 1 but got tensor of shape {}'.format(feat.shape)
        assert feat.shape[2] == hm.shape[2]
        assert feat.shape[3] == hm.shape[3]

        #Apply 1x1 Convolution to features
        if self.conv1x1:
            feat = self.conv1x1_layer(feat)

        #Multiply heatmap with features element-wise
        x = torch.mul(feat, hm)

        #Global max pooling
        x = self.gmp(x)
        x = x.view(x.shape[0], -1)

        #Classification
        score = self.classifier(x)

        return score, x