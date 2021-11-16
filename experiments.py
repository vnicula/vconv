# https://github.com/auspicious3000/autovc
# https://github.com/r9y9/wavenet_vocoder/
# https://github.com/miaoYuanyuan/gen_melSpec_from_wav

# NOTE: for the pretrained wavenet vocoder at https://github.com/auspicious3000/autovc:
#       hparams file says fmin=125,        but actually fmin=90
#       hparams file says ref_level_db=20, but actually ref_level_db=16

import os
import pickle

import numpy as np
import math

import torch
import librosa

import IPython.display as ipd
from matplotlib import pyplot as plt

device = torch.device('cuda:0')

# Modified from https://github.com/auspicious3000/autovc
from synthesis import build_model
from synthesis import wavegen
from hparams import hparams

wavenet_vocoder = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth", map_location=device)
wavenet_vocoder.load_state_dict(checkpoint["state_dict"])
# Modified from https://github.com/auspicious3000/autovc
from model_vc import Generator

def pad_seq(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

auto_VC = Generator(32,256,512,32).eval().to(device)
g_checkpoint = torch.load('autovc.ckpt', map_location=device) 
auto_VC.load_state_dict(g_checkpoint['model'])

metadata = pickle.load(open('metadata.pkl', "rb"))
spkr_to_embed = {entry[0] : entry[1] for entry in metadata}

def apply_autoVC(mel, embed_in, embed_out):
    # assume normalized mel spectrogram as input (normalized to db scale)
    # assume numpy input for both mel spect and embedding
    mel, len_pad = pad_seq(mel)
    
    mel       = torch.from_numpy(      mel[np.newaxis, ...]).to(device)
    embed_in  = torch.from_numpy( embed_in[np.newaxis, ...]).to(device)
    embed_out = torch.from_numpy(embed_out[np.newaxis, ...]).to(device)
    
    with torch.no_grad():
        mel_no_PN, mel_yes_PN, _ = auto_VC(mel, embed_in, embed_out)
            
        if len_pad == 0:
            mel_no_PN  =  mel_no_PN[0, 0, :, :].cpu().numpy()
            mel_yes_PN = mel_yes_PN[0, 0, :, :].cpu().numpy()
        else:
            mel_no_PN  =  mel_no_PN[0, 0, :-len_pad, :].cpu().numpy()
            mel_yes_PN = mel_yes_PN[0, 0, :-len_pad, :].cpu().numpy()
    
    return mel_no_PN, mel_yes_PN

# Modified from https://github.com/miaoYuanyuan/gen_melSpec_from_wav
sr = 16000
n_fft = 1024
win_length = 1024
hop_length = 256
n_mels = 80
fmin = 90
fmax = 7600
ref_level_db = 16
min_level_db = -100

# def visualize_spect(spect, title=None):
#     plt.figure()
#     if title is not None:
#         plt.title(title)
#     plt.imshow(np.flip(spect**.25, 0))
#     plt.show()
    
def _amp_to_db(x):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)
    
def normalize_for_VC(mel):
    # assume magnitude melspectrum with correct sr/fmin/fmax as input
    mel = _amp_to_db(mel) - ref_level_db
    mel = np.clip((mel - min_level_db) / -min_level_db, 0, 1)
    return mel.T

def denormalize_from_VC(mel):
    mel = (np.clip(mel, 0, 1) * -min_level_db) + min_level_db
    mel = _db_to_amp(mel + ref_level_db)
    return mel.T

def perform_experiment(
    audio_in, 
    lin_to_mel = True, 
    mel_to_lin = False, 
    apply_VC   = True, 
    embed_in   = None,
    embed_out  = None,
    apply_PN   = True, 
    inversion  = "WV" # or "GL"
):    
    lin_in = np.abs(librosa.stft(audio_in, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
    
    if lin_to_mel:
        mel_in = librosa.feature.melspectrogram(S=lin_in, sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax, n_mels=n_mels) 
        
    if apply_VC:
        mel_in = normalize_for_VC(mel_in)
        
        mel_no_PN, mel_yes_PN = apply_autoVC(mel_in, embed_in, embed_out)        
        mel_out = mel_yes_PN if apply_PN else mel_no_PN 
    else:
        lin_out = lin_in
        mel_out = mel_in
    
    if inversion == "WV":
        if not apply_VC:
            mel_out = normalize_for_VC(mel_out)
        audio = wavegen(wavenet_vocoder, c=mel_out)
    
    if mel_to_lin:
        if apply_VC:
            mel_out = denormalize_from_VC(mel_out)
        lin_out = librosa.feature.inverse.mel_to_stft(mel_out, n_fft=n_fft, sr=sr, fmin=fmin, fmax=fmax) 
    
    if inversion == "GL":
        audio = librosa.griffinlim(lin_out, win_length=win_length, hop_length=hop_length)
        
    return audio
# an example experiment
audio_in, _ = librosa.load('audio/p225_001.wav', sr)

embed_in = spkr_to_embed['p225']
embed_out = spkr_to_embed['p256']

audio_out = perform_experiment(audio_in, embed_in=embed_in, embed_out=embed_out)

ipd.display(ipd.Audio(audio_in, rate=sr))
ipd.display(ipd.Audio(audio_out, rate=sr))

# all experiments in a loop (only convert between p225 and p256)

def save_and_show(audio, path):
    print(path)
    librosa.output.write_wav(path=path, y=audio, sr=sr)
    ipd.display(ipd.Audio(audio, rate=sr))

for fname in ['audio/p225_001.wav', 'audio/p256_002.wav']:
    spkr_in = fname.split("/")[-1][:4]
    for spkr_out in ['p225', 'p256']:
        audio_in, _ = librosa.load(fname, sr=sr)
        
        embed_in  = spkr_to_embed[spkr_in]
        embed_out = spkr_to_embed[spkr_out]
        
        # A
        audio_out = perform_experiment(
            audio_in,
            lin_to_mel = True,
            apply_VC   = False,
            apply_PN   = False,
            inversion  = "WV"
        )
        path = 'audio/'+spkr_in+'_A_lin_mel_WV.wav'
        save_and_show(audio_out, path)
                
        # B
        audio_out = perform_experiment(
            audio_in,
            apply_VC   = False,
            apply_PN   = False,
            inversion  = "GL"
        )
        path = 'audio/'+spkr_in+'_B_lin_GL.wav'
        save_and_show(audio_out, path)
        
        # C
        audio_out = perform_experiment(
            audio_in,
            lin_to_mel = True,
            mel_to_lin = True,
            apply_VC   = False,
            apply_PN   = False,
            inversion  = "GL"
        )
        path = 'audio/'+spkr_in+'_C_lin_mel_lin_GL.wav'
        save_and_show(audio_out, path)
        
        # D
        audio_out = perform_experiment(
            audio_in,
            lin_to_mel = True,
            apply_VC   = True,
            embed_in   = embed_in,
            embed_out  = embed_out,
            apply_PN   = True,
            inversion  = "WV"
        )
        path = 'audio/'+spkr_in+'_D_'+spkr_out+'_lin_mel_VC_mel_PN_mel_WV.wav'
        save_and_show(audio_out, path)
        
        # E
        audio_out = perform_experiment(
            audio_in,
            lin_to_mel = True,
            apply_VC   = True,
            embed_in   = embed_in,
            embed_out  = embed_out,
            apply_PN   = False,
            inversion  = "WV"
        )
        path = 'audio/'+spkr_in+'_E_'+spkr_out+'_lin_mel_VC_mel_WV.wav'
        save_and_show(audio_out, path)
        
        # F
        audio_out = perform_experiment(
            audio_in,
            lin_to_mel = True,
            apply_VC   = True,
            embed_in   = embed_in,
            embed_out  = embed_out,
            apply_PN   = True,
            mel_to_lin = True,
            inversion  = "GL"
        )
        path = 'audio/'+spkr_in+'_F_'+spkr_out+'_lin_mel_VC_mel_PN_mel_lin_GL.wav'
        save_and_show(audio_out, path)
        
        # G
        audio_out = perform_experiment(
            audio_in,
            lin_to_mel = True,
            apply_VC   = True,
            embed_in   = embed_in,
            embed_out  = embed_out,
            apply_PN   = False,
            mel_to_lin = True,
            inversion  = "GL"
        )
        path = 'audio/'+spkr_in+'_G_'+spkr_out+'_lin_mel_VC_mel_lin_GL.wav'
        save_and_show(audio_out, path)