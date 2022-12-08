
from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        #print('logits',self.logits)
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot[:,0]=1
        return one_hot

    def forward(self, image_q, image_k, add_to_queue):
        self.image_shape = image_q.shape[2:]
        
        self.logits2, self.target2 ,_,_2, self.intermediate= self.model(image_q, image_k, add_to_queue, return_intermediate_outputs=True)
        #self.logits2, self.target2 ,_,_2= self.model(image_q, image_k, add_to_queue)
        self.logits = self.logits2[0]
        return self.logits2, self.target2,_,_2

    def backward(self, ids):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list
   
    def backward(self, ids, target_layer):
        """
        Class-specific backpropagation
        """
        # one hot encode the location of the correct key (0)
        one_hot = self._encode_one_hot(ids)
        fmaps = self.intermediate['layer4']
        #print('fmaps',fmaps.shape) #1*2048*1*7*7
        #print('one_hot',one_hot)
        #print('logits2',self.logits2[0].shape)
        grad_wrt_act = torch.autograd.grad(outputs=self.logits2[0], inputs=fmaps, grad_outputs=one_hot, create_graph=True)[0]

        return grad_wrt_act

    def generate(self, target_layer, grads):
        
        fmaps = self.intermediate['layer4']
        weights = F.adaptive_avg_pool2d(grads, 1) 
        #print('fmaps.shape',fmaps.shape)
        #print('weights.shape',weights.shape)
        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        #print('gcam_shape',gcam.shape)
        B, C, T, H, W = gcam.shape

        gcam_raw = gcam
        gcam_raw = gcam_raw.view(B, -1)
        gcam_raw -= gcam_raw.min(dim=1, keepdim=True)[0]
        gcam_raw /= (gcam_raw.max(dim=1, keepdim=True)[0]+0.0000001)
        gcam_raw = gcam_raw.view(B, C, T, H, W)

        gcam = F.relu(gcam_raw)

        # uncomment to scale gradcam to image size
        # gcam = F.interpolate(
        #     gcam, self.image_shape, mode="bilinear", align_corners=False
        # )
        """
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= (gcam.max(dim=1, keepdim=True)[0]+0.0000001)
        gcam = gcam.view(B, C, T, H, W)
        """
        return gcam
