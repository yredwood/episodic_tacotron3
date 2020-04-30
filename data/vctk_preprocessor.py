import os
from pathlib import Path
import pdb
# generate filelist from vctk root directory
# run in the root

root_dir = 'data-bin/VCTK-Corpus'

wav_folder_name = 'wav48'
txt_folder_name = 'txt'

output_root = 'filelists'

speaker_list = os.listdir(os.path.join(os.path.join(root_dir, txt_folder_name)))
fname_list = list(Path(os.path.join(root_dir, wav_folder_name)).rglob('*.wav'))

wavs, labels, sids = [], [], []
for wav_fname in fname_list:
    if 'p315' in str(wav_fname):
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


num_train_sid = int(0.9 * len(speaker_list))
train_sids = speaker_list[:num_train_sid]
print ('SIDs for training: ', train_sids)
print ('Num tr {} | Num test {}'.format(len(train_sids), len(speaker_list[num_train_sid:])))

train_fname = os.path.join(output_root, 'vctk_train.txt')
val_fname = os.path.join(output_root, 'vctk_val.txt')

train_lines = []
val_lines = []
for i in range(len(sids)):
    line = '{}|{}|{}'.format(
        wavs[i], labels[i], sids[i]
    )
    if sids[i] in train_sids:
        train_lines.append(line)
    else:
        val_lines.append(line)

with open(train_fname, 'wt') as f:
    f.writelines('\n'.join(train_lines))

with open(val_fname, 'wt') as f:
    f.writelines('\n'.join(val_lines))
