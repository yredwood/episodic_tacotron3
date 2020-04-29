import sys
sys.path.append('waveglow/')

import os
from shutil import copyfile
import numpy as np
import scipy as sp
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from audio_processing import griffin_lim
from hparams import create_hparams
from model import load_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from data_utils import EpisodicLoader, EpisodicCollater, EpisodicBatchSampler
from text import cmudict, text_to_sequence, sequence_to_text

import pdb


# ========== parameters ===========
checkpoint_path = 'models/lin_mels_2_test/checkpoint_12000'
waveglow_path = 'models/pretrained/waveglow_256channels_v4.pt'
#waveglow_path = 'models/pretrained/waveglow_46000'
audio_path = 'filelists/libri100_val.txt'
#audio_path = 'filelists/trump.txt'
num_support_save = 4

test_text_list = [
    'AITRICS leads the race to optimized precision care, strengthening and trust.',
    'Our mission is to improve patient outcomes.',
    '"Oh, I believe everything I am told," said the Caterpillar.',
    'I think you must be deceived so far.',
    'Did you cross the bridge at that time?',
    'She did not turn her head towards him, although, having such a long and slender neck,' \
            + 'she could have done so with very little trouble',
]

use_griffin_lim = True

supportset_sid = '2952'  # m
#supportset_sid = '1069' # f 
supportset_sid = '8123' # f 
#supportset_sid = '5001'
output_root = 'audios'

output_dir = os.path.join(
        output_root,
        '-'.join(checkpoint_path.split('/')[-2:]) + '-' + supportset_sid,
)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_figure(mel_pred, attention, fname, description='None'):
    description = description[:30]
    gridsize = (3,1)
    fig = plt.figure(figsize=(20,20))
    ax1 = plt.subplot2grid(gridsize, (0,0))
    ax2 = plt.subplot2grid(gridsize, (1,0), rowspan=2)

    ax1.imshow(mel_pred)
    ax2.imshow(attention)
    ax2.set_title(description)
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def linear_decode(lin_pred):
    taco_stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length, 
            sampling_rate=hparams.sampling_rate)
    mel_decompress = taco_stft.spectral_de_normalize(lin_pred + taco_stft.ref_level_db)
    mel_decompress = mel_decompress.data.cpu()

    waveform = griffin_lim(torch.autograd.Variable(mel_decompress[:,:,:-1]), taco_stft.stft_fn, 120)
    return waveform

hparams = create_hparams()
stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)

# load tacotron2 model
model = load_model(hparams).cuda().eval()
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

# load waveglow model
waveglow = torch.load(waveglow_path)['model'].cuda().eval()
denoiser = Denoiser(waveglow).cuda().eval()

arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')

if hparams.model_name == 'episodic-transformer':
    print ('episodic transformer test')
    valset = EpisodicLoader(audio_path, hparams)
    collate_fn = EpisodicCollater(1, hparams) # frame = 1

    batch_sampler = EpisodicBatchSampler(valset.sid_to_index, hparams, shuffle=False)

    for batch_idx in batch_sampler:
        _, _, sid = valset.audiopaths_and_text[batch_idx[0]]
        if sid != supportset_sid:
            continue
        batch = collate_fn([valset[idx] for idx in batch_idx])
        break
        
    x, y = model.parse_batch(batch.copy())
    for i in range(num_support_save):
        ref_idx = batch['support']['idx'].data.tolist().index(i)
        
        audio_path, test_text, speaker = valset.audiopaths_and_text[batch_idx[i]]

        # 1. save original mel spec
        fname_wav = os.path.join(output_dir, 'ref_true_{}.wav'.format(i))
        lin_spec = x['support']['lin_padded'][ref_idx:ref_idx+1]
        mel_spec = x['support']['mel_padded'][ref_idx:ref_idx+1]
        lin_len = (lin_spec.mean(1) != 0).sum()
        audio = linear_decode(lin_spec[:,:,:lin_len])
        write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
        save_figure(mel_spec[0].data.cpu().numpy(),
                 np.zeros((10,10)), fname_wav.replace('.wav', '.png'),
                 description=test_text)

        # 2. save prediction 
        text_encoded = torch.LongTensor(text_to_sequence(test_text,
            hparams.text_cleaners,
            arpabet_dict))[None,:].cuda()
        text_lengths = torch.LongTensor([len(text_encoded[0])]).cuda()

        input_dict = {
                'query': {
                    'text_padded': text_encoded,
                    'text_length': text_lengths,
                },
                'support': x['support'], 
                'ref_idx': ref_idx,
                }

        with torch.no_grad():
            outputs = model.inference(input_dict)
            audio = linear_decode(outputs['lin_post'])

        fname_wav = os.path.join(output_dir, 'ref_pred_{}.wav'.format(i))
        write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
        save_figure(outputs['mel_post'][0].data.cpu().numpy(),
                outputs['alignments'][0].data.cpu().numpy(), 
                fname_wav.replace('.wav', '.png'), 
                description=test_text)
        print (test_text)

    # save test texts 
    for tidx, test_text in enumerate(test_text_list):
        text_encoded = torch.LongTensor(
                text_to_sequence(test_text,
                    hparams.text_cleaners,
                    arpabet_dict)
                )[None,:].cuda()
        text_lengths = torch.LongTensor(
                [len(text_encoded[0])]).cuda()

        input_dict = {
                'query': {
                    'text_padded': text_encoded,
                    'text_length': text_lengths,
                },
                'support': x['support'], 
                'ref_idx': ref_idx,
                }

        with torch.no_grad():
            outputs = model.inference(input_dict)
            audio = linear_decode(outputs['lin_post'])

        fname_wav = os.path.join(output_dir, '{}.wav'.format(tidx))
        write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
        save_figure(outputs['mel_post'][0].data.cpu().numpy(), 
                outputs['alignments'][0].data.cpu().numpy().T, 
                fname_wav.replace('.wav', '.png'), 
                description=test_text)
        print (test_text)


    




## save reference wavs and aligned predictions
#for idx in range(len(dataloader)):
#    audio_path, text, sid = dataloader.audiopaths_and_text[idx]
#    if sid != supportset_sid:
#        continue
#
#    # save original wav file
#    fname_wav = os.path.join(output_dir, 'ref_true_{}.wav'.format(idx))
#    copyfile(audio_path, fname_wav)
#
#    text_encoded = torch.LongTensor(\
#            text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()
#    mel = load_mel(audio_path)
#
#    # save reconstruction from true mel
#    fname_wav = os.path.join(output_dir, 'ref_recon_{}.wav'.format(idx))
#    with torch.no_grad(): 
#        audio = denoiser(waveglow.infer(mel, sigma=0.8), 0.01)[:,0]
#    write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
#    fname_fig = os.path.join(output_dir, 'ref_mel.png'.format(idx))
#    save_figure(mel[0].data.cpu().numpy(), 
#            np.zeros((10,10)), fname_fig)
#        
#    # save parallel prediction
#    fname_wav = os.path.join(output_dir, 'ref_parallel_{}.wav'.format(idx))
#    with torch.no_grad():
#        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(
#                (text_encoded, mel, None, None))
#        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:,0]
#    write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
#
#    fname_fig = os.path.join(output_dir, 'attention_{}.png'.format(idx))
#    save_figure(mel_outputs_postnet[0].data.cpu().numpy(), 
#            alignments[0].data.cpu().numpy(), fname_fig)
#
#
#    print (idx, text)
#    break
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
##
