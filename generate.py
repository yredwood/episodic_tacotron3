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
checkpoint_path = 'models/episodic_dual/checkpoint_24000'
waveglow_path = 'models/pretrained/waveglow_256channels_v4.pt'
#waveglow_path = 'models/pretrained/waveglow_46000'
audio_path = 'filelists/libri100_val.txt'
num_support_save = 5

test_text_list = [
    'AITRICS leads the race to optimized precision care, strengthening and trust.',
    'Our mission is to improve patient outcomes.',
    '"Oh, I believe everything I am told," said the Caterpillar.',
    'I think you must be deceived so far.',
    'Did you cross the bridge at that time?',
    'She did not turn her head towards him, although, having such a long and slender neck,' \
            + 'she could have done so with very little trouble',
]

use_griffin_lim = False

#supportset_sid = '2952'  # m
#supportset_sid = '1069' # f 
supportset_sid = '8123' # f 
output_root = 'audios'

output_dir = os.path.join(
        output_root,
        '-'.join(checkpoint_path.split('/')[-2:]) + '-' + supportset_sid,
)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_mel(path):
    sampling_rate, audio = read(path)
    audio = torch.from_numpy(audio.astype(np.float32))
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec

def load_dataloader(hparams, audio_path):
    dataloader = TextMelLoader(audio_path, hparams)
    datacollate = TextMelCollate(1)
    return dataloader, datacollate

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

def decode(mel_outputs_postnet):
    if use_griffin_lim:
        taco_stft = TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length, 
                sampling_rate=hparams.sampling_rate)
        mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
        mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling
        waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), 
                taco_stft.stft_fn, 60)
    else:
        waveform = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:,0]

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
        mel_outputs_postnet = x['support'][2][ref_idx:ref_idx+1]
        # remove pad
        mel_len = (mel_outputs_postnet.mean(1) != 0).sum()
        mel_outputs_postnet = mel_outputs_postnet[:,:,:mel_len]
        audio = decode(mel_outputs_postnet)
        write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
        save_figure(mel_outputs_postnet[0].data.cpu().numpy(),
                 np.zeros((10,10)), fname_wav.replace('.wav', '.png'),
                 description=test_text)

        # 2. save prediction 
        text_encoded = torch.LongTensor(text_to_sequence(test_text,
            hparams.text_cleaners,
            arpabet_dict))[None,:].cuda()
        text_lengths = torch.LongTensor([len(text_encoded[0])]).cuda()

        input_dict = {'query': (text_encoded, text_lengths, ref_idx), 
                'support': x['support'], 'ref_idx': ref_idx}
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(input_dict)
            audio = decode(mel_outputs_postnet)

        fname_wav = os.path.join(output_dir, 'ref_pred_{}.wav'.format(i))
        write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
        save_figure(mel_outputs_postnet[0].data.cpu().numpy(),
                alignments[0].data.cpu().numpy(), fname_wav.replace('.wav', '.png'), 
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

        input_dict = {'query': (text_encoded, text_lengths), 
                'support': x['support'], 'ref_idx': ref_idx}

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model.inference(input_dict)
            audio = decode(mel_outputs_postnet)
        
        fname_wav = os.path.join(output_dir, '{}.wav'.format(tidx))
        write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
        save_figure(mel_outputs_postnet[0].data.cpu().numpy(), 
                alignments[0].data.cpu().numpy().T, fname_wav.replace('.wav', '.png'), 
            description=test_text)
        print (test_text)


else:
    print ('gst test')
    # dataloader
    arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
    dataloader, datacollate = load_dataloader(hparams, audio_path)

    cnt = 0
    for idx in range(len(dataloader)):
        audio_path, text, sid = dataloader.audiopaths_and_text[idx]
        if sid != supportset_sid:
            continue

        fname_wav = os.path.join(output_dir, 'ref_true_{}.wav'.format(idx))
        copyfile(audio_path, fname_wav)
        
        # save waveglow original mel
        mel = load_mel(audio_path)

        fname_wav = os.path.join(output_dir, 'ref_recon_{}.wav'.format(idx))
        with torch.no_grad():
            audio = decode(mel)
        write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
        fname_fig = os.path.join(output_dir, 'true_mel_{}.png'.format(idx))
        save_figure(mel[0].data.cpu().numpy(), np.zeros((10,10)), fname_fig, text)


        # save waveglow prediction mel
        fname_wav = os.path.join(output_dir, 'pred_{}.wav'.format(idx))
        text_encoded = torch.LongTensor(\
                text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None,:].cuda().long()
        with torch.no_grad():
            _, mel_post, _, attn = model.inference((text_encoded, mel))
            audio = decode(mel_post)
        write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
        
        fname_fig = os.path.join(output_dir, 'pred_mel_{}.png'.format(idx))
        save_figure(mel_post[0].data.cpu().numpy(), attn[0].data.cpu().numpy(), fname_fig, text)

        print (idx, text)

        # non-parallel predictions
        for text in test_text_list:
            text_encoded = torch.LongTensor(
                text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None,:].cuda().long()
            fname_wav = os.path.join(output_dir, 'ref{}_{}.wav'.format(text, idx))
            with torch.no_grad():
                _, mel_post, _, attn = model.inference((text_encoded, mel))
                audio = decode(mel_post)
            write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
            
            fname_fig = os.path.join(output_dir, 'ref{}_{}.png'.format(text, idx))
            save_figure(mel_post[0].data.cpu().numpy(), attn[0].data.cpu().numpy(), fname_fig, text)
            print (idx, text)

    # without style 
#    for text in test_text_list:
#        text_encoded = torch.LongTensor(
#            text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None,:].cuda().long()
#        fname_wav = os.path.join(output_dir, 'noref_{}.wav'.format(text, idx))
#        with torch.no_grad():
#            mel_post, _ = model.inference((text_encoded, None))
#            audio = denoiser(waveglow.infer(mel_post, sigma=0.8), 0.01)[:,0]
#        write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
#        
#        fname_fig = os.path.join(output_dir, 'ref{}_{}.png'.format(text, idx))
#        save_figure(mel_post[0].data.cpu().numpy(), np.zeros((10,10)), fname_fig, text)
#        print (idx, text)
    
        cnt += 1
        if cnt > num_support_save:
            break

    




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
