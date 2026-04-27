from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import Malignant_Cls_Agent.infer_llnm.models.configs as configs
from Malignant_Cls_Agent.infer_llnm.models.attention import Attention
from Malignant_Cls_Agent.infer_llnm.models.embed import Embeddings 
from Malignant_Cls_Agent.infer_llnm.models.mlp import Mlp
from Malignant_Cls_Agent.infer_llnm.models.block import Block

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if i < 4:
                layer = Block(config, vis, mm=True)  # 前4层需要处理多模态融合
            else:
                layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, text=None):
        attn_weights = []
        
        for (i, layer_block) in enumerate(self.layer):     
            if i == 4:
                hidden_states = torch.cat((hidden_states, text), 1)  
                hidden_states, weights = layer_block(hidden_states)
            elif i < 4:
                hidden_states, text, weights = layer_block(hidden_states, text)       # This layyer for fusion block
            else:
                hidden_states, weights = layer_block(hidden_states)                   # This layyer for self-attention block

            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


