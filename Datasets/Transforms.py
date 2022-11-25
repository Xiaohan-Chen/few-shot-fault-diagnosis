'''
One dimentional signal augmentation.
Device: GPU
Reference: https://github.com/iver56/audiomentations

Data: 2022/11/19
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
'''

import numpy as np
import torch

def probability():
    '''
    return a probability
    '''
    return np.random.random()

def ToTensor(signal):
    '''
    Numpy to Tensor
    '''
    return torch.from_numpy(signal).float()

def AddGaussianSNR(signal):
    '''
    Reference: https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/add_gaussian_snr.py
    Add Gaussian noise to the signal, a random signal-to-noise ratio will be picked.
    '''
    min_snr = 3,  # (int): minimum signal-to-noise ration in dB
    max_snr = 30, # (int): maximum signal-to-noise ration in dB
    signal_length = signal.size(-1)
    device = signal.device

    snr = np.random.randint(min_snr, max_snr, dtype=int)

    clear_rms = torch.sqrt(torch.mean(torch.square(signal)))
    a = float(snr) / 20
    noise_rms = clear_rms / (10**a)
    noise = torch.normal(0.0, noise_rms, size=(signal_length,))

    return signal + noise.to(device)

def Shift(signal):
    '''
    Shift the signal forwards or backwards
    '''
    shift_factor = np.random.uniform(0,0.3)
    shift_length = round(signal.size(-1) * shift_factor)
    num_places_to_shift = round(np.random.uniform(-shift_length, shift_length))
    shifted_signal = torch.roll(signal, num_places_to_shift, dims=-1)
    return shifted_signal

def TimeMask(signal):
    '''
    Randomly mask a part of signal
    '''
    signal_length = signal.size(-1)
    device = signal.device
    signal_copy = signal.clone()

    mask_factor = np.random.uniform(0.1,0.45)
    max_mask_length = int(signal_length * mask_factor)  # maximum mask band length
    mask_length = np.random.randint(max_mask_length) # randomly choose a mask band length
    mask_start = np.random.randint(0, signal_length) # randomly choose a mask band start point

    while mask_start + mask_length > max_mask_length:
        mask_start = np.random.randint(0, signal_length)
    mask = torch.zeros(mask_length, dtype=float, device=device)

    # signal transformations are only apply for a singal signal, not considering batch_size dimention
    if len(signal.size()) == 1:
        signal_copy[mask_start:mask_start+mask_length] *= mask
    elif len(signal.size()) == 2:
        signal_copy[:, mask_start:mask_start+mask_length] *= mask
    else:
        raise Exception("{} dimentinal signal time masking is not implemented, please try 1 or 2 dimentional signal.")
    
    return signal_copy

def Fade(signal):
    '''
    Reference: https://pytorch.org/audio/stable/generated/torchaudio.transforms.Fade.html#torchaudio.transforms.Fade
    Add a fade in and/or fade out to signal
    '''
    signal_length = signal.size(-1)
    fade_in_length = int(0.3 * signal_length)
    fade_out_length = int(0.3 * signal_length)

    device = signal.device
    signal_copy = signal.clone()

    # fade in
    if probability() > 0.5:
        fade_in = torch.linspace(0,1,fade_in_length,device=device)
        fade_in = torch.log10(0.1 + fade_in) + 1 # logarithmic
        ones = torch.ones(signal_length - fade_in_length, device=device)
        fade_in_mask = torch.cat((fade_in, ones))
        signal_copy *= fade_in_mask
    
    # fade out
    if probability() > 0.5:
        fade_out = torch.linspace(1,0,fade_out_length,device=device)
        fade_out = torch.log10(0.1 + fade_out) + 1 # logarithmic
        ones = torch.ones(signal_length - fade_out_length, device=device)
        fade_out_mask = torch.cat((ones, fade_out))
        signal_copy *= fade_out_mask
    
    return signal_copy


def Gain(signal):
    '''
    Reference: https://pytorch.org/audio/stable/generated/torchaudio.transforms.Vol.html#torchaudio.transforms.Vol
    Multiply the audio by a random amplitude factor to reduce or increase the volume.
    '''
    gain_min = 0.5
    gain_max = 1.5
    gain_factor = np.random.uniform(gain_min, gain_max)

    signal_copy = signal.clone()

    return signal_copy * gain_factor

def Reverse(signal):
    '''
    Horizontally reverse the signal: start <--> end
    '''
    if len(signal.size()) == 1:
        signal_copy = torch.flip(signal, dims=[0])
    elif len(signal.size()) == 2:
        signal_copy = torch.flip(signal, dims=[1])
    else:
        raise Exception("{} dimentinal signal time masking is not implemented, please try 1 or 2 dimentional signal.")
    
    return signal_copy

def Normal(signal):
    '''
    0-1 normalization
    '''
    return (signal - signal.min()) / (signal.max() - signal.min())

def random_waveform_transforms(signal):
    '''
    Random waveform signal transformation
    '''
    # move to GPU
    signal = ToTensor(signal).float()

    # 70% probability to add Gaussian noise
    if probability() > 0.3:
        signal = AddGaussianSNR(signal)

    # 40% probability to shift the signal
    if probability() > 0.6:
        signal = Shift(signal)

    # 50% probability to mask time
    if probability() > 0.5:
        signal = TimeMask(signal)
    
    # 50% probability to fade
    if probability() > 0.5:
        signal = Fade(signal)

    # 40% probability to gain
    if probability() > 0.6:
        signal = Gain(signal)

    # 30% probability to reverse
    if probability() > 0.7:
        signal = Reverse(signal)
    
    signal = Normal(signal)

    return signal

def waveform_transforms_test(signal):
    '''
    Random waveform signal transformation
    '''
    # move to GPU
    signal = ToTensor(signal).float()
    signal = Normal(signal)
    return signal


def random_spectrogram_transforms(signal):
    pass