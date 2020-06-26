import os
import subprocess
import sagemaker
import torch
from sagemaker import get_execution_role
from helper.utils import *
import argparse
import librosa
import torch
from scipy.signal import lfilter
import numpy as np
import scipy.io.wavfile
import soundfile as sf
import fnmatch, os, warnings
import json

sample_rate=16000

def wavread(filename):
    fs, x = scipy.io.wavfile.read(filename)
    if np.issubdtype(x.dtype, np.integer):
        x = x / np.iinfo(x.dtype).max
#     print(fs,x)
    return x, fs
def write(inputs, filename, sr=sample_rate):
    librosa.output.write_wav(filename, inputs, sr)
    
def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: # [B, C, T]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results

# def wavwrite(filename, s, fs):
#     if s.dtype != np.int16:
#         s = np.array(s * 2**15, dtype=np.int16)
#     if np.any(s > np.iinfo(np.int16).max) or np.any(s < np.iinfo(np.int16).min):
#         warnings.warn('Warning: clipping detected when writing {}'.format(filename))
#     scipy.io.wavfile.write(filename, fs, s)

def bin_interp(upcount, lwcount, upthr, lwthr, margin, tol=0.1):
    n_iter = 1
    if abs(upcount - upthr - margin) < tol:
        midcount = upcount
    elif abs(lwcount - lwthr - margin) < tol:
        midcount = lwcount
    else:
        midcount = (upcount + lwcount)/2
        midthr = (upthr + lwthr)/2
        diff = midcount - midthr - margin
        while abs(diff) > tol:
            n_iter += 1
            if n_iter > 20:
                tol *= 1.1
            if diff > tol:
                midcount = (upcount + midcount)/2
                midthr = (upthr + midthr)/2
            elif diff < -tol:
                midcount = (lwcount + midcount)/2
                midthr = (lwthr + midthr)/2
            diff = midcount - midthr - margin
    return midcount

def asl_meter(x, fs, nbits=16):
    '''Measure the Active Speech Level (ASR) of x following ITU-T P.56.
    If x is integer, it will be scaled to (-1, 1) according to nbits.
    '''

    if np.issubdtype(x.dtype, np.integer):
        x = x / 2**(nbits-1)

    # Constants
    MIN_LOG_OFFSET = 1e-20
    T = 0.03                # Time constant of smoothing in seconds
    g = np.exp(-1/(T*fs))
    H = 0.20                # Time of handover in seconds
    I = int(np.ceil(H*fs))
    M = 15.9                # Margin between threshold and ASL in dB

    a = np.zeros(nbits-1)                       # Activity count
    c = 0.5**np.arange(nbits-1, 0, step=-1)     # Threshold level
    h = np.ones(nbits)*I                        # Hangover count
    s = 0
    sq = 0
    p = 0
    q = 0
    asl = -100

    L = len(x)
    s = sum(abs(x))
    sq = sum(x**2)
    dclevel = s/np.arange(1, L+1)
    lond_term_level = 10*np.log10(sq/np.arange(1, L+1) + MIN_LOG_OFFSET)
    c_dB = 20*np.log10(c)

    for i in range(L):
        p = g * p + (1-g) * abs(x[i])
        q = g * q + (1-g) * p

        for j in range(nbits-1):
            if q >= c[j]:
                a[j] += 1
                h[j] = 0
            elif h[j] < I:
                a[j] += 1
                h[j] += 1

    a_dB = -100 * np.ones(nbits-1)

    for i in range(nbits-1):
        if a[i] != 0:
            a_dB[i] = 10*np.log10(sq/a[i])

    delta = a_dB - c_dB
    idx = np.where(delta <= M)[0]

    if len(idx) != 0:
        idx = idx[0]
        if idx > 1:
            asl = bin_interp(a_dB[idx], a_dB[idx-1], c_dB[idx], c_dB[idx-1], M)
        else:
            asl = a_dB[idx]

    return asl

def rms_energy(x):
    return 10*np.log10((1e-12 + x.dot(x))/len(x))

def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def separate_process(x, predictor):
    ilens = np.array([x.shape[0]])
    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(x).float()], pad_value)
    ilens = torch.from_numpy(ilens) 
    mixture_np = mixtures_pad.numpy()
    mix_lengths = ilens.cpu()
    # Forward
    response = predictor.predict(mixture_np)
    response = torch.from_numpy(response) 
    estimate_source = response.cpu()
    # Remove padding and flat
    flat_estimate = remove_pad(estimate_source, mix_lengths)
    return flat_estimate[0]

def preprocess_wav(filename):
    x, fs = wavread(filename)
    x_mirror = x
    asl_level = -26.0

    y = x + x_mirror
    y = y/10**(asl_meter(y, fs)/20) * 10**(asl_level/20)
    return x, fs
