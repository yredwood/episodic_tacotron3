import librosa
import os
import pdb

from joblib import Parallel, delayed

from pathlib import Path
import numpy as np
from scipy.io.wavfile import read, write
from tqdm import tqdm
import sys

def write_single(wav_fname, top_db, resample_rate, output_folder):
    data, sample_rate = librosa.load(wav_fname, sr=None)
    trimmed, _ = librosa.effects.trim(data, top_db=top_db)

    resampled = librosa.core.resample(trimmed, sample_rate, resample_rate)
    y = (resampled * 32767.0).astype(np.int16)

    # exclude utterances > 10.sec
    length = y.shape[0]
    if length > resample_rate * 10.:
        return None

    target_fname = os.path.join(output_wav_folder, str(wav_fname).split('/')[-1])
    write(target_fname, resample_rate, y)
    return y.shape[0] / float(resample_rate)
    

if __name__ == '__main__':

    root_dir = sys.argv[1] # data root (VCTK/VCTK-Corpus/)
    assert root_dir
    wav_folder_name = 'wav48'
    txt_folder_name = 'txt'

    output_root = './filelists'

    resample_rate = 24000
    top_db = 20 # trimming threshold
    print ('trimming db: ',top_db)
    # top_db is set to have median duration to 1.8s as in https://arxiv.org/pdf/1806.04558.pdf
    # but doubt if its well trimmed
    output_wav_folder = os.path.join(root_dir, 'wav{}k_{}'.format(resample_rate//1000, top_db))

    if not os.path.exists(output_wav_folder):
        os.makedirs(output_wav_folder)

    speaker_list = os.listdir(os.path.join(os.path.join(root_dir, txt_folder_name)))
    fname_list = list(Path(os.path.join(root_dir, wav_folder_name)).rglob('*.wav'))
    
    wavs, labels, sids = [], [], []
    for wav_fname in fname_list:
        if 'p315' in str(wav_fname):
            # found error
            continue

        _sid = str(wav_fname).split('/')[-2]
        _label_fname = str(wav_fname).replace(wav_folder_name, txt_folder_name)
        _label_fname = _label_fname.replace('.wav', '.txt')

        with open(_label_fname, 'rt') as f:
            lines = f.readlines()
            assert len(lines) == 1

        _label = lines[0].strip()
        wavs.append(str(wav_fname))
        labels.append(_label)
        sids.append(_sid)
    
    # debug
#    wavs = wavs[:3000]
#    sids = sids[:3000]
#    labels = labels[:3000]

    lengths = Parallel(n_jobs=20)(
            delayed(write_single)(wav_fname, top_db, resample_rate, output_wav_folder) for wav_fname in tqdm(wavs)
    )
    exclude_idx = [i for i in range(len(lengths)) if lengths[i] is None]
    lengths = [lengths[i] for i in range(len(lengths)) if i not in exclude_idx]
    wavs =  [wavs[i] for i in range(len(wavs)) if i not in exclude_idx]
    labels =  [labels[i] for i in range(len(labels)) if i not in exclude_idx]
    sids =  [sids[i] for i in range(len(sids)) if i not in exclude_idx]

    print ('Median length (in sec): {:.3f}'.format(sorted(lengths)[len(lengths)//2]))

    # writing files
    num_train_sid = int(0.9 * len(speaker_list))
    train_sids = speaker_list[:num_train_sid]
    print ('training SIDs: ', train_sids)
    print ('Num tr {} | Num test {}'.format(num_train_sid, len(speaker_list) - num_train_sid))

    train_fname = os.path.join(output_root, 'vctk_train.txt')
    val_fname = os.path.join(output_root, 'vctk_val.txt')

    train_lines, val_lines = [], []
    for i in range(len(sids)):

        target_fname = os.path.join(output_wav_folder, str(wavs[i]).split('/')[-1])
        if not os.path.exists(target_fname):
            continue
        line = '{}|{}|{}'.format(
                target_fname, labels[i], sids[i]
        )
        if sids[i] in train_sids:
            train_lines.append(line)
        else:
            val_lines.append(line)

    with open(train_fname, 'wt') as f:
        f.writelines('\n'.join(train_lines))

    with open(val_fname, 'wt') as f:
        f.writelines('\n'.join(val_lines))
