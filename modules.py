# adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
# MIT License
#
# Copyright (c) 2018 MagicGirl Sakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import PackedSequence

import pdb

class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hp):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
                          hidden_size=hp.ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = hp.n_mel_channels
        self.ref_enc_gru_size = hp.ref_enc_gru_size

    def forward(self, inputs):
        out = inputs.transpose(-1,-2)
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, token_embedding_size//2]
    '''
    def __init__(self, hp):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.token_embedding_size // hp.num_heads))
        #d_q = hp.token_embedding_size // 2
        d_q = hp.ref_enc_gru_size
        d_k = hp.token_embedding_size // hp.num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=hp.token_embedding_size,
            num_heads=hp.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class _GST(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder = ReferenceEncoder(hp)
        #self.stl = STL(hp)
        self.stl = nn.Linear(hp.ref_enc_gru_size, hp.token_embedding_size)

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        #style_embed = self.stl(enc_out)
        style_embed = self.stl(enc_out).unsqueeze(1)

        return style_embed


class TransformerStyleTokenLayer(nn.Module):
    def __init__(self, hp):
        super().__init__()
#        self.encoder = ReferenceEncoder(hp)
        #self.stl = STL(hp)
#        self.stl = nn.Linear(hp.ref_enc_gru_size, hp.token_embedding_size)

        self.gst = GST(hp)

#        self.sentence_encoder = nn.GRU(input_size=hp.encoder_embedding_dim,
#                hidden_size=hp.sentence_encoder_dim, batch_first=True)
#
#        self.mab = MAB_qkv(hp.sentence_encoder_dim,
#                hp.sentence_encoder_dim,
#                hp.token_embedding_size, 
#                hp.token_embedding_size, num_heads=1)

    def forward(self, text, text_len, rtext, rtext_len, rmel):
        style_embed = self.gst(rmel)
        return style_embed

    def _forward(self, text, text_len, rtext, rtext_len, rmel):
        style_embed = self.gst(rmel) # bsz_s,1,token_embed_size
        style_embed.transpose(0,1).repeat(text.size(0),1,1)

        self.sentence_encoder.flatten_parameters()
        text_len = text_len.cpu().numpy()
        _tp = nn.utils.rnn.pack_padded_sequence(text, text_len, batch_first=True)
        _, query = self.sentence_encoder(_tp) # 1, bsz, encoder_embedding_dim

        rtext_len = rtext_len.cpu().numpy()
        _rtp = nn.utils.rnn.pack_padded_sequence(rtext, rtext_len, batch_first=True)
        _, key = self.sentence_encoder(_rtp) # 1, bsz_s, encoder_embedding_dim

        style, attn = self.mab(query.transpose(0,1), key.repeat(query.size(1),1,1),
                style_embed, get_attn=True)
        attn = attn.reshape(style.size(0), -1, attn.size(-1)).mean(1)
        entropy = torch.log(attn).mean()

        #print (attn.squeeze().data.cpu().numpy())
        print ('NENT: {:.4f} | TEMP: {:.4f}'.format(-entropy.data.cpu().numpy(),
            self.mab.T.item()))
        print (attn.argmax(-1).squeeze().data.cpu().numpy())

#        Q = query.transpose(0,1)
#        K = key.repeat(text.size(0), 1, 1).transpose(1,2)
#        mattn = (Q@K)/math.sqrt(Q.size(-1))
#        pdb.set_trace()
        
        # bsz, 1, token_embedding_size
        return style

class MAB_qkv(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim, num_heads=8, ln=False, p=None):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_q, dim)
        self.fc_k = nn.Linear(dim_k, dim)
        self.fc_v = nn.Linear(dim_v, dim)
        self.fc_o = nn.Linear(dim, dim)
        self.T = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.T, 10.)

    def forward(self, query, key, value, get_attn=False):
        Q, K, V = self.fc_q(query), self.fc_k(key), self.fc_v(value)
        A_logits = Q @ K.transpose(-2,-1) / math.sqrt(query.size(-1)) * self.T
        #A_logits = query @ key.transpose(-2,-1) / math.sqrt(query.size(-1)) * self.T
        A = torch.softmax(A_logits, -1)
        attn = A @ value
        out = self.fc_o(attn)
        if get_attn:
            return out, A
        return out


class _MAB_qkv(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim, num_heads=8, ln=False, p=None):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_q, dim)
        self.fc_k = nn.Linear(dim_k, dim)
        self.fc_v = nn.Linear(dim_v, dim)
        self.fc_o = nn.Linear(dim, dim)

        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.dropout1 = nn.Dropout(p=p) if p is not None else nn.Identity()
        self.dropout2 = nn.Dropout(p=p) if p is not None else nn.Identity()


    def forward(self, query, key, value, mask=None, get_attn=False):
        Q, K, V = self.fc_q(query), self.fc_k(key), self.fc_v(value)
        Q_ = torch.cat(Q.chunk(self.num_heads, -1), 0)
        K_ = torch.cat(K.chunk(self.num_heads, -1), 0)
        V_ = torch.cat(V.chunk(self.num_heads, -1), 0)

        A_logits = (Q_ @ K_.transpose(-2, -1)) /  math.sqrt(Q.shape[-1]) * 1.
        if mask is not None:
            mask = torch.stack([mask]*Q.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, 0)
            A_logits.masked_fill_(mask, -float('inf'))
            A = torch.softmax(A_logits, -1)
            # to prevent underflow due to no attention
            A.masked_fill_(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        
#        attn = torch.cat((A @ V_).chunk(self.num_heads, 0), -1)
#        O = self.ln1(Q + self.dropout1(attn))
#        O = self.ln2(O + self.dropout2(F.relu(self.fc_o(O))))
#        if get_attn:
#            return O, A
#        return O
        attn = torch.cat((A @ V_).chunk(self.num_heads, 0), -1)
        O = self.fc_o(attn)
        if get_attn:
            return O, A
        return O



class Block(nn.Module):
    def __init__(self, n_in, n_out, shortcut=False):
        super().__init__()

        filter_size = 3
        padding = filter_size // 2
        self.conv1 = nn.Conv2d(n_in, n_out, filter_size, stride=2, padding=padding, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(n_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class ConvEmbedding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.inner_dim = 32
        self.input_dim = 80
        
        layers = [Block(1, 32), Block(32, 32)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(-1,-2)
        x = x.view(x.size(0), 1, -1, self.input_dim)
        x = self.backbone(x) # b, 32, T//4, 80//4
        x = x.permute(0,2,1,3)
        x = x.reshape(x.size(0), x.size(1), -1) # b,T//4, 32*20
        x = x.transpose(0,1)
        return x


class LSTM_BN(nn.Module):
    def __init__(self, input_size, hidden_size, shortcut, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']
        self.hidden_size = hidden_size 
        self.shortcut = shortcut

        self.rnn_list = nn.ModuleList([
            nn.LSTM(input_size if layers==0 else hidden_size*2,
                hidden_size, num_layers=1, bidirectional=True)
            for layers in range(self.num_layers)])

        self.dense_list = nn.ModuleList([
            nn.Linear(self.hidden_size*2, self.hidden_size*2)
            for layer in range(self.num_layers-1)])

        self.bn_list = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_size*2) 
            for layer in range(self.num_layers-1)])

    def forward(self, input, hidden):
        h, c = hidden
        def reshape_hidden(x):
            nl2, bsz, hdim = x.size()
            x = x.reshape(self.num_layers,2,bsz,hdim)
            return x
        h = reshape_hidden(h)
        c = reshape_hidden(c)

        hlist, clist = [], []
        for i, rnn in enumerate(self.rnn_list):
            residual = input.data
            output, (ho, co) = rnn(input, (h[i], c[i]))
            hlist.append(ho)
            clist.append(co)
            
            input = output.data
            if i!=self.num_layers-1:
                input = self.dense_list[i](input)
                input = self.bn_list[i](input)
                input = F.relu(input)

            if i > 0 and self.shortcut:
                input = input + residual

            input = get_packed_sequence(
                    data=input, batch_sizes=output.batch_sizes,
                    sorted_indices=output.sorted_indices,
                    unsorted_indices=output.unsorted_indices)

        hlist = torch.cat(hlist, 0)
        clist = torch.cat(clist, 0)
        return input, (hlist, clist)


class GST(nn.Module):
    def __init__(self, hp):
        '''
        las-style encoder
        '''
        super().__init__()

        input_dim = hp.n_mel_channels
        self.pre_conv = ConvEmbedding(input_dim)
        self.preconv_dim = 32 * self.pooling(input_dim)
        
        self.lstm_hidden_dim = (hp.token_embedding_size + hp.speaker_embedding_dim) // 2 
        self.lstm_num_layers = 4
        self.lstm = LSTM_BN(
                input_size=self.preconv_dim,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_num_layers,
                dropout=0.1,
                bidirectional=True,
                shortcut=True,
        )
        self.output_dim = self.lstm_hidden_dim * 2


    def forward(self, x):
        
        x_len = self.calc_length(x)
        x_emb = self.pre_conv(x) # (T,bsz,640)
        #x_emb = F.dropout(x_emb, p=0.5, training=self.training)

        pooled_length = [self.pooling(_l) for _l in x_len]
        pooled_length = x_emb.new_tensor(pooled_length).long()
        #assert pooled_length[0] == x_emb.size(0)

        state_size = self.lstm_num_layers*2, x_emb.size(1), self.lstm_hidden_dim
        fw_x = nn.utils.rnn.pack_padded_sequence(x_emb, pooled_length, enforce_sorted=False)
        fw_h = x_emb.new_zeros(*state_size)
        fw_c = x_emb.new_zeros(*state_size)
        packed_outputs, (final_hiddens, final_cells) = self.lstm(fw_x, (fw_h, fw_c))

        # not using final_h, final_c
#        final_outs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=.0)
#        #final_outs = F.dropout(final_outs, p=0.5, training=self.training)
#        return final_outs.transpose(0,1), pooled_length

        # using final_h
        final_outs = final_hiddens.view(-1,2,final_hiddens.size(-2),final_hiddens.size(-1))
        final_outs = torch.cat((final_outs[-1,0], final_outs[-1,1]), dim=-1)

        return final_outs.unsqueeze(1)


        

    def pooling(self, x):
        for _ in range(len(self.pre_conv.backbone)):
            #x = (x - 3 + 2 * 3//2) // 2 + 1
            x = x // 2 
        return x

    def calc_length(self, x):
        x_len = [x.size(-1) for _ in range(x.size(0))]
        for t in reversed(range(x.size(-1))):
            pads = (x[:,:,t].sum(1) == 0).int().tolist()
            x_len = [x_len[i] - pads[i] for i in range(len(x_len))]

            if sum(pads) == 0:
                break
        return x_len



def get_packed_sequence(data, batch_sizes, sorted_indices, unsorted_indices):
        return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)







        #
