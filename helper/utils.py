import math

import argparse

import librosa
import torch

from scipy.signal import lfilter
import numpy as np
import scipy.io.wavfile
import soundfile as sf

import fnmatch, os, warnings


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


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


def write(inputs, filename, sr=16000):
    librosa.output.write_wav(filename, inputs, sr)

def wavread(filename):
    fs, x = scipy.io.wavfile.read(filename)
    if np.issubdtype(x.dtype, np.integer):
        x = x / np.iinfo(x.dtype).max
    return x, fs

def wavwrite(filename, s, fs):
    if s.dtype != np.int16:
        s = np.array(s * 2**15, dtype=np.int16)
    if np.any(s > np.iinfo(np.int16).max) or np.any(s < np.iinfo(np.int16).min):
        warnings.warn('Warning: clipping detected when writing {}'.format(filename))
    scipy.io.wavfile.write(filename, fs, s)

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


def rms_energy(x):
    return 10*np.log10((1e-12 + x.dot(x))/len(x))

def preprocess_wav(filename):
    # filename = './1-20.wav'
    x, fs = wavread(filename)
    x_mirror = x
    # x = np.float32(x)
    # print(x.dtype)
    # N_dB = rms_energy(x_mirror)
    # S_dB = asl_meter(x, fs)
    # print(N_dB)
    # print(S_dB)

    # N_new = S_dB
    # x_mirror = 10**(N_new/20) * x_mirror / 10**(N_dB/20)
    asl_level = -26.0

    y = x + x_mirror
    y = y/10**(asl_meter(y, fs)/20) * 10**(asl_level/20)
    return x, fs

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
    mix_lengths = ilens.cuda()
    # Forward
    response = predictor.predict(mixture_np)
#     print(response)
#     estimate_source = response.argmax(axis=1)[0]
    response = torch.from_numpy(response) 
    estimate_source = response.cuda()
    # Remove padding and flat
#     print(estimate_source)
    flat_estimate = remove_pad(estimate_source, mix_lengths)

#     with torch.no_grad():
#         mixture, mix_lengths = mixtures_pad.cuda(), ilens.cuda()

#         # Forward
#         estimate_source = model(mixture)  # [B, C, T]
        
#         # Remove padding and flat
#         flat_estimate = remove_pad(estimate_source, mix_lengths)
    return flat_estimate[0]

if __name__ == '__main__':
    torch.manual_seed(123)
    M, C, K, N = 2, 2, 3, 4
    frame_step = 2
    signal = torch.randint(5, (M, C, K, N))
    result = overlap_and_add(signal, frame_step)
    print(signal)
    print(result)
