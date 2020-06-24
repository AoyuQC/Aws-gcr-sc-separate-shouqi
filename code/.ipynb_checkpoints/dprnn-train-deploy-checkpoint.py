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


import argparse
import os
import sys
import textwrap

import torch
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder

import numpy as np
import torch
from six import BytesIO
import yaml

###############################
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
from utils import remove_pad
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from asteroid.data import LibriMix
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import DPRNNTasNet

import json
import argparse

from utils import remove_pad
#############################


# def input_fn(request_body, request_content_type):
#     """An input_fn that loads a pickled tensor"""
#     if request_content_type == 'application/python-pickle':
#         eval_dataset = EvalDataset(in_dir, out_dir+'mix.json',  # 修改
#                                    batch_size=1,
#                                    sample_rate=16000)
#         eval_loader = EvalDataLoader(eval_dataset, batch_size=1)

#         return eval_loader
#     else:
#         pass

def model_fn(model_dir):
    with open(os.path.join(model_dir, 'best_model.pth'), 'rb') as f:
        model = DPRNNTasNet.from_pretrained(f)
    return model


def predict_fn(input_data, model):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.eval()
    model.cuda()
    with torch.no_grad():
        # Forward
        estimate_source = model(input_data)  # [B, C, T]
        # mixture, mix_lengths, filenames = data
        # mixture, mix_lengths = mixture, mix_lengths  # .cuda()
        # # Forward
        # estimate_source = model(mixture)  # [B, C, T]
        # # Remove padding and flat
        # flat_estimate = remove_pad(estimate_source, mix_lengths)
        # mixture = remove_pad(mixture, mix_lengths)
        # size = flat_estimate[0].shape  # 默认只有一个sample
        # ept = np.zeros((size[0]+1, size[1]))
        # ept[0, :] = mixture[0]
        # ept[1:, :] = flat_estimate[0]
        # for (i, data) in enumerate(input_data):  # eval_loader
        #     mixture, mix_lengths, filenames = data
        #     mixture, mix_lengths = mixture, mix_lengths  # .cuda()
        #     # Forward
        #     estimate_source = model(mixture)  # [B, C, T]
        #     # Remove padding and flat
        #     flat_estimate = remove_pad(estimate_source, mix_lengths)
        #     mixture = remove_pad(mixture, mix_lengths)
        #     size = flat_estimate[0].shape  # 默认只有一个sample
        #     ept = np.zeros((size[0]+1, size[1]))
        #     ept[0, :] = mixture[0]
        #     ept[1:, :] = flat_estimate[0]

    return estimate_source


def _train(args):
#     train_dir = args.train
#     val_dir = args.test

#     with open('conf.yml') as f:
#             def_conf = yaml.safe_load(f)

#     parser = prepare_parser_from_dict(def_conf, parser=parser)
#     arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
#     print(arg_dic)
#     conf = arg_dic

#     train_set = WhamDataset(train_dir, conf['data']['task'],
#                             sample_rate=conf['data']['sample_rate'], segment=conf['data']['segment'],
#                             nondefault_nsrc=conf['data']['nondefault_nsrc'])
#     val_set = WhamDataset(val_dir, conf['data']['task'], segment=conf['data']['segment'],
#                           sample_rate=conf['data']['sample_rate'], nondefault_nsrc=conf['data']['nondefault_nsrc'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=conf['training']['batch_size'],
                            num_workers=conf['training']['num_workers'],
                            drop_last=True)
    # Update number of source values (It depends on the task)
    conf['masknet'].update({'n_src': train_set.n_src})

    model = DPRNNTasNet(**conf['filterbank'], **conf['masknet'])
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    # Define scheduler
    scheduler = None
    if conf['training']['half_lr']:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5,
                                      patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    # exp_dir = conf['main_args']['exp_dir']
    # os.makedirs(exp_dir, exist_ok=True)
    exp_dir = args.model_dir
    conf_path = os.path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    system = System(model=model, loss_func=loss_func, optimizer=optimizer,
                    train_loader=train_loader, val_loader=val_loader,
                    scheduler=scheduler, config=conf)
    system.batch_size = 1

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min', save_top_k=5, verbose=1)
    early_stopping = False
    if conf['training']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                       verbose=1)

    # Don't ask GPU if they are not available.
    # print("!!!!!!!{}".format(torch.cuda.is_available()))
    # print(torch.__version__)
    gpus = -1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(max_epochs=conf['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         early_stop_callback=early_stopping,
                         default_root_dir=exp_dir,
                         gpus=gpus,
                         distributed_backend='ddp',
                         gradient_clip_val=conf['training']["gradient_clipping"])
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # Save best model (next PL version will make this easier)
    best_path = [b for b, v in best_k.items() if v == min(best_k.values())][0]
    state_dict = torch.load(best_path)
    system.load_state_dict(state_dict=state_dict['state_dict'])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, 'best_model.pth'))
    # return _save_model(model, model_dir)

# def _save_model(model, model_dir):
#     logger.info("Saving the model.")
#     path = os.path.join(model_dir, 'model.pth')
#     # recommended way from http://pytorch.org/docs/master/notes/serialization.html
#     torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str,
                        default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    _train(parser.parse_args())

    # ... load from args.train and args.test, train a model, write model to args.model_dir.
