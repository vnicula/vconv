import argparse
from collections import OrderedDict
from math import ceil
from pathlib import Path
from typing import List, Tuple
import librosa
import math
import matplotlib.pylab as plt
import os
import numpy as np
import soundfile as sf
import pickle
import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.nn.functional import normalize, pad

# import wav2mel
from model_bl import D_VECTOR
from model_vc import Generator
from mel_spect import get_mel_spect

from parallel_wavegan.utils import load_model
from parallel_wavegan.bin.preprocess import logmelfilterbank

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

def chunks(lst: List, n: int) -> List[List]:
    for i in range(0, len(lst), n):
        yield lst[i : (i + n)]


def pad_seq(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def get_embed(encoder: nn.Module, mel: Tensor) -> Tensor:
    emb = encoder(mel[None, :])
    return emb


def save_spect(save_path, spec):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(spec.T, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.savefig(save_path)

    return fig


def get_speaker_model():
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256)
    c_checkpoint = torch.load('3000000-BL.ckpt', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    # C.remove_weight_norm()
    C = C.eval()
    return C


def load_vocoder(vocoder_path):
    # load vocoder
    vocoder = load_model(vocoder_path)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval()
    # print('vocoder', vocoder)
    return vocoder    


def load_autovc(vc_path):
    g_checkpoint = torch.load(vc_path, map_location='cpu')
    G = Generator(32,256,512,32)
    G.load_state_dict(g_checkpoint['model'])
    # G.remove_weight_norm()
    G = G.eval()
    return G

def spect_for_vc(audio_in):
    lin_in = np.abs(librosa.stft(audio_in, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
    mel_in = librosa.feature.melspectrogram(S=lin_in, sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax, n_mels=n_mels) 
    mel_in = normalize_for_VC(mel_in)

    return mel_in

def mel_to_wav_grif(mel_out):
    mel_out = denormalize_from_VC(mel_out)
    lin_out = librosa.feature.inverse.mel_to_stft(mel_out, n_fft=n_fft, sr=sr, fmin=fmin, fmax=fmax) 
    audio = librosa.griffinlim(lin_out, win_length=win_length, hop_length=hop_length)
        
    return audio

def apply_autoVC(auto_VC, mel, embed_in, embed_out, device):
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

def main(
    model_path: Path,
    vocoder_path: Path,
    source: Path,
    target: Path,
    output: Path,
):

    metadata = pickle.load(open('metadata.pkl', "rb"))
    spkr_to_embed = {entry[0] : entry[1] for entry in metadata}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    # model = torch.jit.load(model_path).to(device)
    # vocoder = torch.jit.load(vocoder_path).to(device)
    model = load_autovc(model_path).to(device)
    vocoder = load_vocoder(vocoder_path).to(device)
    speaker_encoder = get_speaker_model().to(device)
    # print(vocoder)

    src_sr = sr
    src, _ = librosa.load(source, sr=src_sr)
    print('src shape, sr:', src.shape, src_sr)

    src_mel = spect_for_vc(src)
    print('src_mel shape:', src_mel.shape)

    # src_mel = torch.from_numpy(get_mel_spect(src)).to(device)
    # tgt_mel = torch.from_numpy(get_mel_spect(tgt)).to(device)

    # src_mel_spk = wav2mel(src, src_sr).to(device)
    # tgt_mel_spk = wav2mel.Wav2Mel()(tgt, tgt_sr).to(device)

    # src_mel = torch.from_numpy(logmelfilterbank(
    #     src.view(-1).numpy(), src_sr, fft_size=2048, hop_size=200, win_length=800, fmin=50)).to(device)
    # tgt_mel = torch.from_numpy(logmelfilterbank(
    #     tgt.view(-1).numpy(), tgt_sr, fft_size=2048, hop_size=200, win_length=800, fmin=50)).to(device)

    # save_spect(os.path.splitext(target)[0]+'.jpg', tgt_mel.cpu())

    # src_emb = get_embed(speaker_encoder, src_mel)
    # tgt_emb = get_embed(speaker_encoder, tgt_mel)
    src_emb = spkr_to_embed['p225']
    tgt_emb = spkr_to_embed['p228']

    # src_mel, len_pad = pad_seq(src_mel)
    save_spect(os.path.splitext(source)[0]+'.jpg', src_mel)
    # src_mel = src_mel[None, :]

    mel_no_PN, mel = apply_autoVC(model, src_mel, src_emb, tgt_emb, device)
    # with torch.no_grad():
    #     _, mel, _ = model(src_mel, src_emb, tgt_emb)
    # # mel = src_mel
    # mel = mel[0, :, :] if len_pad == 0 else mel[0, :-len_pad, :]

    print('Converted mel shape: ', mel.shape)
    save_spect(os.path.splitext(output)[0]+'.jpg', mel)

    # mel = (mel - 0.8) / 0.2

    mel = denormalize_from_VC(mel)
    print('Denormalized mel shape: ', mel.shape)

    with torch.no_grad():
        # wav = vocoder.generate([mel])[0].data.cpu().numpy()
        wav = vocoder.inference(c=denormalize_from_VC(src_mel).T, normalize_before=False)
        print('Wave shape: ', wav.shape)
        wav = wav.view(-1).cpu().numpy()
    
    # wav = mel_to_wav_grif(mel)
    sf.write(output, wav.astype(np.float32), sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("vocoder_path", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("output", type=Path)
    main(**vars(parser.parse_args()))
