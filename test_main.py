"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).

This is the callback (non-blocking) version.
"""

import json
import yaml
import librosa
import matplotlib.pyplot as plt
# from numba import jit
import numpy as np
import pyaudio
import time
# from librosa.filters import mel as librosa_mel_fn
# from librosa.util import normalize
import torch
# import torchaudio
import torch.utils.data

from mel_spect import get_mel_spect
# import synthesis
from model_bl import D_VECTOR
from collections import OrderedDict
from math import ceil
from model_vc import Generator

from parallel_wavegan.utils import load_model

CHANNELS = 1

attr_d = {
    "segment_size": 8000, #8192,
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,

    "sampling_rate": 16000,

    "fmin": 0,
    "fmax": 8000,
    "fmax_for_loss": 0,
}

# to_mel = torchaudio.transforms.MelSpectrogram(
#     n_mels=80, n_fft=1024, win_length=1024, hop_length=256)
# mean, std = -4, 4

def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
):
    """Compute log-Mel filterbank feature.
    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
        log_base (float): Log base. If set to None, use np.log.
    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).
    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))

    if log_base is None:
        return np.log(mel)
    elif log_base == 10.0:
        return np.log10(mel)
    elif log_base == 2.0:
        return np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def get_speaker_model():
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load('3000000-BL.ckpt')
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    return C

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

# config = 'singlevc/pretrained/HiFi-GAN/UNIVERSAL_V1/config.json'
# # config = 'config_v3.json'
# with open(config) as f:
#     data = f.read()

# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self

# json_config = json.loads(data)
# h = AttrDict(json_config)
# print(h)

# Load HiFi GAN
# torch_checkpoints = torch.load("singlevc/pretrained/HiFi-GAN/UNIVERSAL_V1/g_02500000", map_location=torch.device('cpu'))
# # torch_checkpoints = torch.load("generator_v3", map_location=torch.device('cpu'))
# torch_generator_weights = torch_checkpoints["generator"]
# torch_model = Generator(h)
# torch_model.load_state_dict(torch_checkpoints["generator"])
# torch_model.eval()
# torch_model.remove_weight_norm()


#prepare models
def load_vocoder():
    # load vocoder
    # vocoder = load_model("Vocoder/checkpoint-400000steps.pkl").to('cuda').eval()
    # vocoder = load_model("Vocoder/checkpoint-400000steps.pkl").eval()
    vocoder = load_model("pretrained_model/arctic_slt_parallel_wavegan.v1/checkpoint-400000steps.pkl").cuda()
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().cuda()
    print('vocoder', vocoder)
    return vocoder

# vocoder = synthesis.build_model().cuda()
# checkpoint = torch.load("checkpoint_step001000000_ema.pth")
# vocoder.load_state_dict(checkpoint["state_dict"])
# vocoder = load_vocoder()

def load_autovc():
    device = 'cuda:0'
    G = Generator(32,256,512,32).eval().to(device)

    g_checkpoint = torch.load('autovc.ckpt', map_location=device)
    G.load_state_dict(g_checkpoint['model'])
    return G

conv = load_autovc()
spke = get_speaker_model()
vocoder = load_vocoder()

def callback(in_data, frame_count, time_info, status):
    data = np.frombuffer(in_data, dtype=np.float32)
    
    spec = get_mel_spect(data)
    # spec = preprocess(data)
    # print(spec[:10])
    print('mel shape:', spec.shape)
    spec, pad_len = pad_seq(spec)
    print('pad_len:', pad_len)
    melsp = torch.from_numpy(spec[np.newaxis, :, :]).cuda()
    with torch.no_grad():
        emb = spke(melsp)
        _, mel, _ = conv(melsp, emb, emb)
    #     spec = SVCGen.infer(spec)
    #     print('trans shape:', spec.shape)
    #     hifigan_output = torch_model(spec)
        mel = mel[0, :, :] if pad_len == 0 else mel[0, :-pad_len, :]
        mel = mel.squeeze()
        # c = x_identic_psnt.squeeze() #.to('cuda')
        print(emb.shape, melsp.shape)
        y_out = vocoder.inference(c=melsp.squeeze()).view(-1).cpu().numpy()
        print('y_out shape:', y_out.shape)
        # y_out = y_out.view(-1).cpu().numpy()

    # waveform = synthesis.wavegen(vocoder, c=spec)
    # output = hifigan_output.squeeze().detach().numpy()
    

    # return (y_out[:attr_d["segment_size"]], pyaudio.paContinue)
    return (y_out[:attr_d["segment_size"]], pyaudio.paContinue)


stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=attr_d["sampling_rate"],
                # rate=24000,
                input=True,
                output=True,
                frames_per_buffer=attr_d["segment_size"],
                # frames_per_buffer=24000,
                stream_callback=callback)

print("Starting to listen.")
stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

stream.stop_stream()
stream.close()

p.terminate()
