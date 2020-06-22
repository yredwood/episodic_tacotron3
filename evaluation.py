import sys
sys.path.append('waveglow/')

import os
import argparse
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

from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

import pdb


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


def decode(mel, hparams, vocoder=None):
    # model = (denoiser, waveglow_model)
    if vocoder is None:
        # use griffin lim
        taco_stft = TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length, 
                sampling_rate=hparams.sampling_rate)
        mel_decompress = taco_stft.spectral_de_normalize(mel)
        mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling
        waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), 
                taco_stft.stft_fn, 60)
    else:
        waveform = vocoder[0](vocoder[1].infer(mel, sigma=0.8), 0.01)[:,0]
    return waveform

def get_speaker_info(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    out = {} 
    for line in lines:
        sid, gender = line.split('|')
        out[sid] = gender
    return out


def evaluate(args):
    hparams = create_hparams(args.hparam_string)
    hparams.episodic_training = True
    stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

    # load model
    model = load_model(hparams).cuda().eval()
    model.load_state_dict(torch.load(args.ckpt)['state_dict'])
    print ('loaded model from {}'.format(args.ckpt))

    # load vocoder model
    if args.vocoder is not 'griffin_lim':
        waveglow = torch.load(args.vocoder)['model'].cuda().eval()
        denoiser = Denoiser(waveglow).cuda().eval()
        vocoder = (denoiser, waveglow)
    else:
        # use griffin_lim
        vocoder = None
    
    # load dictionary and datasets
    arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')

    dataset = EpisodicLoader(args.audio_path, hparams)
    collate_fn = EpisodicCollater(1, hparams) # 1 is the frame / not implemented for multi-frame prediction
    batch_sampler = EpisodicBatchSampler(dataset.sid_to_index, hparams, shuffle=False)

    # load speaker info
    sinfo = get_speaker_info(args.speaker_info)

    # get target sid
    for batch_idx in batch_sampler:
        _, _, sid = dataset.audiopaths_and_text[batch_idx[0]]
        if sid != args.supportset_sid:
            continue
        batch = collate_fn([dataset[idx] for idx in batch_idx])

    # generate batch, which is model dependent
    test_input, ref_mels, tar_mels = model.get_test_batch(batch.copy(), ref_ind=-1)

    # save reference mel-spec
    print ('get reference mel-specs...')
    ref_audios = []
    tar_audios = []
    for i in range(len(test_input)):
        refmel = ref_mels[i]
        tarmel = tar_mels[i]
        with torch.no_grad():
            raudio = decode(refmel, hparams, vocoder)
            taudio = decode(tarmel, hparams, vocoder)

        ref_audios.append(raudio)
        tar_audios.append(taudio)
        
        if args.save_audio:
            fname_wav = os.path.join(args.output_dir, 'ref-{}.wav'.format(i))
            write(fname_wav, hparams.sampling_rate, raudio[0].data.cpu().numpy())
            save_figure(refmel[0].data.cpu().numpy(), np.zeros((10,10)),
                    fname_wav.replace('.wav', '.png'))

            fname_wav = os.path.join(args.output_dir, 'true-{}.wav'.format(i))
            write(fname_wav, hparams.sampling_rate, taudio[0].data.cpu().numpy())
            save_figure(tarmel[0].data.cpu().numpy(), np.zeros((10,10)),
                    fname_wav.replace('.wav', '.png'))
    



    print ('get predictions...')
    # get prediction and save it
    pred_audios = []
    for i in range(len(test_input)):
        with torch.no_grad():
            _, mel_pred, _, alignments = model.inference(test_input[i])
            audio = decode(mel_pred, hparams, vocoder)
        pred_audios.append(audio)
        
        if args.save_audio:
            fname_wav = os.path.join(args.output_dir, 'pred-{}.wav'.format(i))
            write(fname_wav, hparams.sampling_rate, audio[0].data.cpu().numpy())
            save_figure(mel_pred[0].data.cpu().numpy(), 
                    alignments[0].data.cpu().numpy(),
                    fname_wav.replace('.wav', '.png'))

    output_strings = []
    # compare set of references with set of preds
    encoder = VoiceEncoder()
    def get_embed(fname):
        _w = preprocess_wav(fname)
        return encoder.embed_utterance(_w)
        
    ref_embeds = [get_embed(ad.squeeze(0).data.cpu().numpy()) for ad in ref_audios]
    tar_embeds = [get_embed(ad.squeeze(0).data.cpu().numpy()) for ad in tar_audios]
    pred_embeds = [get_embed(ad.squeeze(0).data.cpu().numpy()) for ad in pred_audios]

    GT = cos_sim(ref_embeds, tar_embeds)
    PR = cos_sim(ref_embeds, pred_embeds)
    output_strings.append('Ground Truth: {:.4f}'.format(np.mean(GT)))
    output_strings.append('Model Prediction: {:.4f}'.format(np.mean(PR)))

    # ======= compare to other sids =========
    same_gender_audios = []

    def get_gender_audios(num_instance, num_people, gender_same):
        '''
        '''
        visited = []
        output_audios = []
        for batch_idx in batch_sampler:
            _, _, sid = dataset.audiopaths_and_text[batch_idx[0]]
            if (sid == args.supportset_sid) and (sid in visited):
                continue

            if (sinfo[sid] != sinfo[args.supportset_sid]) == gender_same:
                continue

            batch = collate_fn([dataset[idx] for idx in batch_idx])
            test_input, ref_mels, tar_mels = model.get_test_batch(batch.copy(), ref_ind=-1)
            
            # get 4 samples from each of other speakers
            for i in range(num_instance):
                commel = tar_mels[i]
                with torch.no_grad():
                    caudio = decode(commel, hparams, vocoder)
                output_audios.append(caudio)

            visited.append(sid)
            if len(visited) == num_people:
                return output_audios

    same_gender = get_gender_audios(8, 4, True)
    diff_gender = get_gender_audios(8, 4, False)

    same_gender_embeds = [get_embed(ad.squeeze(0).data.cpu().numpy()) for ad in same_gender]
    diff_gender_embeds = [get_embed(ad.squeeze(0).data.cpu().numpy()) for ad in diff_gender]

    osg = cos_sim(ref_embeds, same_gender_embeds)
    odg = cos_sim(ref_embeds, diff_gender_embeds)
    output_strings.append('Other same gender: {:.4f}'.format(np.mean(osg)))
    output_strings.append('Other diff gender: {:.4f}'.format(np.mean(odg)))


    # ========== consistency test ==============
    consistency_gt = cos_sim(ref_embeds, ref_embeds)
    consistency_pr = cos_sim(pred_embeds, pred_embeds)
    
    def mean_wo_diag(mat):
        _sum = np.sum(mat) - np.sum(np.diag(mat))
        return _sum / (mat.shape[0] ** 2 - mat.shape[0])
    
    output_strings.append('Consistency GT: {:.4f}'.format(mean_wo_diag(consistency_gt)))
    output_strings.append('Consistency PR: {:.4f}'.format(mean_wo_diag(consistency_pr)))


    # save the result
    print ('\n'.join(output_strings))
    fname = os.path.join(args.output_dir, 'result.txt')
    with open(fname, 'w') as f:
        f.writelines('\n'.join(output_strings))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='models/vctk_gst_pretrained_2gpu/checkpoint_250000')
    parser.add_argument('--vocoder', default='models/waveglow_vctk24k/waveglow_14000',
            help='choices: [waveglow_model_path, griffin_lim')
    parser.add_argument('--audio_path', default='filelists/vctk_24val.txt')
    parser.add_argument('--supportset_sid', default='p287')
    parser.add_argument('--speaker_info', default='filelists/vctk_speaker_info.txt')
    parser.add_argument('--hparam_string', default='num_common=0')
    parser.add_argument('--output_dir', default='audios/test')
    parser.add_argument('--save_audio', type=int, default=1)
    parser.add_argument('--seed', type=int, required=True)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print ('results will be saved in {}'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    evaluate(args)
