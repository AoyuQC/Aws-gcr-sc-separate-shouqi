# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import textwrap
from data import EvalDataLoader, EvalDataset

import torch
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder

import numpy as np
import torch
from six import BytesIO


###############################
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
from utils import remove_pad
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import sys
import os
import torch
from asteroid.data import LibriMix
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid import ConvTasNet
from asteroid.models import ConvTasNet
from asteroid.models import DPRNNTasNet

from data import AudioDataLoader, AudioDataset
import json
import argparse

from conv_tasnet import ConvTasNet
from utils import remove_pad
#############################


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/python-pickle':
        eval_dataset = EvalDataset(in_dir, out_dir+'mix.json', ######修改
                               batch_size=1,
                               sample_rate=16000)
        eval_loader =  EvalDataLoader(eval_dataset, batch_size=1)

        return eval_loader
    else:
        pass

def model_fn(model_dir):    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model = DPRNNTasNet.from_pretrained(f) #
    return model

def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (i, data) in enumerate(input_data): #eval_loader
            mixture, mix_lengths, filenames = data
            mixture, mix_lengths = mixture, mix_lengths#.cuda()
            # Forward
            estimate_source = model(mixture)  # [B, C, T]
            # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)
            mixture = remove_pad(mixture, mix_lengths)
            size = flat_estimate[0].shape #默认只有一个sample
            ept = np.zeros((size[0]+1,size[1]))
            ept[0,:]=mixture[0]
            ept[1:,:]=flat_estimate[0]
            
        return ept 
    
