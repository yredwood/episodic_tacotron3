import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from math import sqrt

from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from model import Prenet, Attention, Tacotron2, Encoder, Postnet
from modules import LSTM_BN, GST
from loss_function import EpisodicLoss
from logger import DualAttentionLogger
import pdb

#class Block(nn.Module):
#    def __init__(self, n_in, n_out, shortcut=False):
#        super().__init__()
#
#        filter_size = 3
#        padding = filter_size // 2
#        self.conv1 = nn.Conv2d(n_in, n_out, filter_size, stride=2, padding=padding, bias=False)
#        self.relu = nn.ReLU()
#        self.bn1 = nn.BatchNorm2d(n_out)
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        return x
#
#class ConvEmbedding(nn.Module):
#    def __init__(self, input_dim):
#        super().__init__()
#        self.inner_dim = 32
#        self.input_dim = 80
#        
#        layers = [Block(1, 32), Block(32, 32)]
#        self.backbone = nn.Sequential(*layers)
#
#    def forward(self, x):
#        x = x.transpose(-1,-2)
#        x = x.view(x.size(0), 1, -1, self.input_dim)
#        x = self.backbone(x) # b, 32, T//4, 80//4
#        x = x.permute(0,2,1,3)
#        x = x.reshape(x.size(0), x.size(1), -1) # b,T//4, 32*20
#        x = x.transpose(0,1)
#        return x
#
#class RefmelEncoder(nn.Module):
#    def __init__(self, hp):
#        '''
#        las-style encoder
#        '''
#        super().__init__()
#
#        input_dim = hp.n_mel_channels
#        self.pre_conv = ConvEmbedding(input_dim)
#        self.preconv_dim = 32 * self.pooling(input_dim)
#        
#        self.lstm_hidden_dim = (hp.ref_embedding_dim) // 2 
#        self.lstm_num_layers = 4
#        self.lstm = LSTM_BN(
#                input_size=self.preconv_dim,
#                hidden_size=self.lstm_hidden_dim,
#                num_layers=self.lstm_num_layers,
#                dropout=0.1,
#                bidirectional=True,
#                shortcut=True,
#        )
#        self.output_dim = self.lstm_hidden_dim * 2
#
#    def forward(self, x, x_len=None):
#        
#        if x_len is None:
#            x_len = self.calc_length(x)
#        x_emb = self.pre_conv(x) # (T,bsz,640)
#        #x_emb = F.dropout(x_emb, p=0.5, training=self.training)
#
#        pooled_length = [self.pooling(_l) for _l in x_len]
#        pooled_length = x_emb.new_tensor(pooled_length).long()
#        #assert pooled_length[0] == x_emb.size(0)
#
#        state_size = self.lstm_num_layers*2, x_emb.size(1), self.lstm_hidden_dim
#        fw_x = nn.utils.rnn.pack_padded_sequence(x_emb, pooled_length, enforce_sorted=False)
#        fw_h = x_emb.new_zeros(*state_size)
#        fw_c = x_emb.new_zeros(*state_size)
#        packed_outputs, (final_hiddens, final_cells) = self.lstm(fw_x, (fw_h, fw_c))
#
##        # not using final_h, final_c
##        final_outs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=.0)
##        final_outs = F.dropout(final_outs, p=0.5, training=self.training)
##        return final_outs.transpose(0,1), pooled_length
#
#        final_outs = final_hiddens.view(-1,2,final_hiddens.size(-2), final_hiddens.size(-1))
#        final_outs = torch.cat((final_outs[-1,0], final_outs[-1,1]),dim=-1)
#        return final_outs.unsqueeze(1)
#
#
#    def pooling(self, x):
#        for _ in range(len(self.pre_conv.backbone)):
#            #x = (x - 3 + 2 * 3//2) // 2 + 1
#            x = x // 2 
#        return x
#
#    def calc_length(self, x):
#        x_len = [x.size(-1) for _ in range(x.size(0))]
#        for t in reversed(range(x.size(-1))):
#            pads = (x[:,:,t].sum(1) == 0).int().tolist()
#            x_len = [x_len[i] - pads[i] for i in range(len(x_len))]
#
#            if sum(pads) == 0:
#                break
#        return x_len
#
#def get_packed_sequence(data, batch_sizes, sorted_indices, unsorted_indices):
#        return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)


class DualAttnDecoder(nn.Module):
    def __init__(self, hparams):
        '''
        encoder_embedding_dim: text encoding
        ref_embedding_dim: reference mel encoding dim: -> new hparam needed
        '''
        super(DualAttnDecoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.text_embedding_dim = hparams.encoder_embedding_dim + hparams.ref_embedding_dim + 128
        self.refmel_embedding_dim = hparams.ref_embedding_dim

        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.p_attention_dropout = hparams.p_attention_dropout

        self.gate_threshold = hparams.gate_threshold
        self.p_teacher_forcing = hparams.p_teacher_forcing


        self.prenet_f0 = ConvNorm(
            1, hparams.prenet_f0_dim,
            kernel_size=hparams.prenet_f0_kernel_size,
            padding=max(0, int(hparams.prenet_f0_kernel_size/2)),
            bias=False, stride=1, dilation=1)

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.text_embedding_dim + 1,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.text_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

#        self.attention_rnn_refmel = nn.LSTMCell(
#            hparams.prenet_dim + self.refmel_embedding_dim,
#            hparams.attention_rnn_dim)
#
#        self.attention_layer_refmel = Attention(
#            hparams.attention_rnn_dim, self.refmel_embedding_dim,
#            hparams.attention_dim, hparams.attention_location_n_filters,
#            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.text_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.text_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.text_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')


    def initialize_rnn(self, mem, mem_length, mem_fn):
        bsz, t, emb_dim = mem.size()
        
        out = {}
        out['mask'] = ~get_mask_from_lengths(mem_length)
        out['attn_h'] = Variable(mem.data.new(bsz, self.attention_rnn_dim).zero_())
        out['attn_c'] = Variable(mem.data.new(bsz, self.attention_rnn_dim).zero_())

        out['attn_w'] = Variable(mem.data.new(bsz, t).zero_())
        out['attn_w_cum'] = Variable(mem.data.new(bsz, t).zero_())
        out['attn_context'] = Variable(mem.data.new(bsz, emb_dim).zero_())

        out['mem'] = mem
        out['mem_processed'] = mem_fn(mem)
        return out

    def initialize_decoder(self, mem):
        bsz, t, emb_dim = mem.size()
        out = {}
        out['dec_h'] = Variable(mem.data.new(bsz, self.decoder_rnn_dim).zero_())
        out['dec_c'] = Variable(mem.data.new(bsz, self.decoder_rnn_dim).zero_())
        return out

    def _decode_single_attn(self, frame, chv, attn_prernn_fn, attn_fn):
        cell_input = torch.cat((frame, chv['attn_context']), -1)
        attn_h, attn_c = attn_prernn_fn(cell_input, (chv['attn_h'], chv['attn_c']))
        attn_h = F.dropout(attn_h, self.p_attention_dropout, self.training)
        attn_c = F.dropout(attn_c, self.p_attention_dropout, self.training)

        attn_w_cat = torch.cat(
            (chv['attn_w'].unsqueeze(1), chv['attn_w_cum'].unsqueeze(1)), dim=1)

        attn_context, attn_w = attn_fn(
                attn_h, chv['mem'], 
                chv['mem_processed'], attn_w_cat, chv['mask']
        )

        # update params
        chv['attn_h'] = attn_h
        chv['attn_c'] = attn_c
        chv['attn_w'] = attn_w
        chv['attn_w_cum'] = attn_w + chv['attn_w_cum']
        chv['attn_context'] = attn_context

        decoder_input = torch.cat(
            (attn_h, attn_context), dim=-1)

        return decoder_input

    def decode(self, input_frame, text_vars, refmel_vars, dec_vars):
        '''
        it will change the dictionary values in vars
        which is hidden and cell states of rnns
        '''

        # add dummy f0 frame
        dummy = input_frame.new_zeros(input_frame.size(0), 1)
        input_frame = torch.cat((input_frame, dummy), dim=1)

        input_text = self._decode_single_attn(input_frame, text_vars,
                self.attention_rnn, self.attention_layer)
#        input_refmel = self._decode_single_attn(input_frame, refmel_vars,
#                self.attention_rnn_refmel, self.attention_layer_refmel)
#        decoder_input = torch.cat(
#            (input_text, input_refmel), dim=-1)

        decoder_input = input_text

        dh, dc = self.decoder_rnn(
                decoder_input, (dec_vars['dec_h'], dec_vars['dec_c']))
        dh = F.dropout(dh, self.p_decoder_dropout, self.training)
        dc = F.dropout(dc, self.p_decoder_dropout, self.training)
        
        # update
        dec_vars['dec_h'] = dh
        dec_vars['dec_c'] = dc

        #context = torch.cat((dh, text_vars['attn_context'], refmel_vars['attn_context']), dim=1)
        context = torch.cat((dh, text_vars['attn_context']), dim=1)
        pred = self.linear_projection(context)
        gate_pred = self.gate_layer(context)
        # attention will be saved in {}_vars['attn_w']
        return pred, gate_pred


    def forward(self, input_dict):

        # 0. unsqueeze dict inputs for clarity
        text_embedding      = input_dict['text_embedding'] # (bsz, t_text, text_embedding_dim)
        text_length         = input_dict['text_length'] # (bsz, t_text)
        refmel_embedding    = input_dict['refmel_embedding'] # (bsz, t_mel, refmel_embedding_dim)
        refmel_length       = input_dict['refmel_length'] # (bsz, t_mel)
        targets             = input_dict['targets'] # (bsz,n_mel_channels,t_mel_true)

        # 1. get teacher forcing inputs (with start frame)
        zero_frame = Variable(text_embedding.data.new(
            1, text_embedding.size(0), self.n_mel_channels * self.n_frames_per_step).zero_())
        tf_inputs = torch.cat((zero_frame, targets.permute(2,0,1)), dim=0)
        tf_inputs = self.prenet(tf_inputs)

        # 2. initialize rnn layers
        # : text_rnn_hc, refmel_rnn_hc, attn_w, attn_context (output)
        # : decoder_rnn
        # -> internal states will be changed inside of the functions
        text_vars = self.initialize_rnn(text_embedding, text_length, self.attention_layer.memory_layer)
        #refmel_vars = self.initialize_rnn(refmel_embedding, refmel_length, self.attention_layer_refmel.memory_layer)
        dec_vars = self.initialize_decoder(text_embedding)

        # 3. autoregressive generation
        mel_outputs, gate_outputs, align_text, align_refmel = [], [], [], []

        for t_frame in range(tf_inputs.size(0)-1):
            if t_frame==0 or np.random.uniform(0.0, 1.0) <= self.p_teacher_forcing:
                rnn_input = tf_inputs[t_frame]
            else:
                rnn_input = self.prenet(mel_outputs[-1])

            mel_output, gate_output = self.decode(
                    rnn_input, text_vars, None, dec_vars)
            attn_t = text_vars['attn_w'] 
            #attn_m = refmel_vars['attn_w']
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            align_text += [attn_t]
            #align_refmel += [attn_m]

        # 4. reshape outputs 
        mel_outputs = torch.stack(mel_outputs).permute(1,2,0) # (t,b,c -> b,c,t)
        gate_outputs = torch.stack(gate_outputs).transpose(0,1) # (t,b -> b,t)
        #align_refmel = torch.stack(align_refmel).transpose(0,1) # (t,b,tr -> b,t,tr)
        align_text = torch.stack(align_text).transpose(0,1) # (t,b,tt -> b,t,tt)
        return mel_outputs, gate_outputs, None, align_text

    def inference(self, input_dict):
        # 0. unsqueeze dict inputs for clarity
        text_embedding      = input_dict['text_embedding'] # (bsz, t_text, text_embedding_dim)
        text_length         = input_dict['text_length'] # (bsz, t_text)
        refmel_embedding    = input_dict['refmel_embedding'] # (bsz, t_mel, refmel_embedding_dim)
        refmel_length       = input_dict['refmel_length'] # (bsz, t_mel)
        targets             = input_dict['targets'] # (bsz,n_mel_channels,t_mel_true)

        # 1. get zero frame
        zero_frame = Variable(
                text_embedding.data.new(
                    text_embedding.size(0), self.n_mel_channels * self.n_frames_per_step
                ).zero_()
        )
        
        # 2. initialize rnn layers
        text_vars = self.initialize_rnn(text_embedding, text_length, self.attention_layer.memory_layer)
        dec_vars = self.initialize_decoder(text_embedding)

        # 3. generation
        rnn_input = zero_frame
        mel_outputs, gate_outputs, align_text, align_refmel = [], [], [], []
        for t_frame in range(self.max_decoder_steps):
            mel_output, gate_output = self.decode(
                    self.prenet(rnn_input), text_vars, None, dec_vars)
            attn_t = text_vars['attn_w']
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            align_text += [attn_t]
            
            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps - 1:
                print ('warning: reached max decoder steps')
                break
            rnn_input = mel_output
        
        # 4. reshape outputs
        mel_outputs = torch.stack(mel_outputs).permute(1,2,0) # (t,b,c -> b,c,t)
        #gate_outputs = torch.stack(gate_outputs).transpose(0,1) # (t,b -> b,t)
        #align_refmel = torch.stack(align_refmel).transpose(0,1) # (t,b,tr -> b,t,tr)
        align_text = torch.stack(align_text).transpose(0,1) # (t,b,tt -> b,t,tt)
        return mel_outputs, None, None, align_text


class DualAttention(nn.Module):
    def __init__(self, hparams):
        super(DualAttention, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams) # text encoder
        #self.gst = RefmelEncoder(hparams)
        self.decoder = DualAttnDecoder(hparams)
        self.postnet = Postnet(hparams)
        self.gst = GST(hparams)
        
        print ('dual attention inited')

    def parse_batch(self, batch):

        q_text_padded     = to_gpu(batch['query']['text_padded']).long()
        q_text_length     = to_gpu(batch['query']['input_lengths']).long()
        q_mel_padded      = to_gpu(batch['query']['mel_padded']).float()
        q_mel_length      = to_gpu(batch['query']['output_lengths']).long()
        q_gate_padded     = to_gpu(batch['query']['gate_padded']).float()

        s_text_padded     = to_gpu(batch['support']['text_padded']).long()
        s_text_length     = to_gpu(batch['support']['input_lengths']).long()
        s_mel_padded      = to_gpu(batch['support']['mel_padded']).float()
        s_mel_length      = to_gpu(batch['support']['output_lengths']).long()
        s_gate_padded     = to_gpu(batch['support']['gate_padded']).float()

        y = {
            'mel': q_mel_padded,
            'gate': q_gate_padded,
            'style': None,
        }
        x = {
            'query': {
                'text_padded': q_text_padded,
                'text_length': q_text_length,
                'mel_padded': q_mel_padded,
                'mel_length': q_mel_length,
            },
            'support': {
                'text_padded': s_text_padded,
                'text_length': s_text_length,
                'mel_padded': s_mel_padded,
                'mel_length': s_mel_length,
            },
        }

        return (x, y)

    def _masked_output(self, x, mask, value=0.0):
        if self.mask_padding and mask is not None:
            # x.size(): bsz, dim, t
            #mask = ~get_mask_from_lengths(x_length) # x.size(0),x.size(-1): bsz, t
            mask = mask.view(x.size(0),1,x.size(-1))
            mask = mask.expand(x.size(0),x.size(1),x.size(-1))
            x.data.masked_fill_(mask, value)
        return x

    def forward(self, inputs):
        support_set = inputs['support']
        query_set = inputs['query']

        query_text_embedding = self.embedding(query_set['text_padded']).transpose(1,2)
        query_text_embedding = self.encoder(query_text_embedding, query_set['text_length'].data)
        
        #refmel_embedding = self.gst(support_set['mel_padded'], support_set['mel_length'].data)
        refmel_embedding = self.gst(support_set['mel_padded'])
        speaker_embedding = refmel_embedding.new_zeros(refmel_embedding.size(0),
                query_text_embedding.size(1), 128)
        # (bsz, 1, refmel_dim) 
        # (bsz, t_mel, refmel_dim) 
        catted = torch.cat(
                (
                    query_text_embedding, 
                    refmel_embedding.repeat(1,query_text_embedding.size(1),1),
                    speaker_embedding,
                ), 
                dim=2)
        
        decoder_inputs = {
            #'text_embedding': query_text_embedding,
            'text_embedding': catted, 
            'text_length': query_set['text_length'].data,
            'refmel_embedding': refmel_embedding, 
            'refmel_length': None,
            'targets': query_set['mel_padded'],
        }

        mel_outputs, gate_outputs, align_refmel, align_text = self.decoder(decoder_inputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        mask = ~get_mask_from_lengths(query_set['mel_length'].data)
        output = {
            'mel': self._masked_output(mel_outputs, mask),
            'mel_post': self._masked_output(mel_outputs_postnet, mask),
            'gate': self._masked_output(gate_outputs.unsqueeze(1), mask, value=1e3).squeeze(1),
            'attn_text': align_text,
            'attn_refmel': align_refmel,
        }

        return output

    def inference(self, inputs):
        support_set = inputs['support']
        query_set = inputs['query']

        query_text_embedding = self.embedding(query_set['text_padded']).transpose(1,2)
        query_text_embedding = self.encoder(query_text_embedding, query_set['text_length'].data)

        #refmel_embedding = self.gst(support_set['mel_padded'], support_set['mel_length'].data)
        refmel_embedding = self.gst(support_set['mel_padded'])
        speaker_embedding = refmel_embedding.new_zeros(refmel_embedding.size(0),
                query_text_embedding.size(1), 128)
        # (bsz, 1, refmel_dim) 
        # (bsz, t_mel, refmel_dim) 
        catted = torch.cat(
                (
                    query_text_embedding, 
                    refmel_embedding.repeat(1,query_text_embedding.size(1),1),
                    speaker_embedding,
                ), 
                dim=2)
        
        decoder_inputs = {
            #'text_embedding': query_text_embedding,
            'text_embedding': catted, 
            'text_length': query_set['text_length'].data,
            'refmel_embedding': refmel_embedding, 
            'refmel_length': None,
            'targets': None,
        }

        mel_outputs, gate_outputs, align_refmel, align_text = self.decoder.inference(decoder_inputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs, mel_outputs_postnet, gate_outputs, align_text
        

    def get_criterion(self, hparams):
        print ('only support episodic loss')
        return EpisodicLoss()

    def get_logger(self, logdir, hparams):
        print ('only support DualAttentionLogger')
        return DualAttentionLogger(logdir)


    def get_test_batch(self, batch, ref_ind=-1):
        # only accepts the same reference for prediction
        assert ref_ind < 0
        q_text_padded     = to_gpu(batch['query']['text_padded']).long()
        q_text_length     = to_gpu(batch['query']['input_lengths']).long()
        q_mel_padded      = to_gpu(batch['query']['mel_padded']).float()
        q_mel_length      = to_gpu(batch['query']['output_lengths']).long()

        s_text_padded     = to_gpu(batch['support']['text_padded']).long()
        s_text_length     = to_gpu(batch['support']['input_lengths']).long()
        s_mel_padded      = to_gpu(batch['support']['mel_padded']).float()
        s_mel_length      = to_gpu(batch['support']['output_lengths']).long()
        
        batches, refmels, tarmels = [], [], []
        for i in range(q_text_padded.size(0)):
            x = {
                'query': {
                    'text_padded': q_text_padded[i:i+1][:,:q_text_length[i]],
                    'text_length': q_text_length[i:i+1].data,
                },
                'support': {
                    'mel_padded': s_mel_padded[i:i+1][:,:,:s_mel_length[i]],
                    'mel_length': s_mel_length[i:i+1],
                },
            }
            batches.append(x)
            refmels.append(s_mel_padded[i:i+1][:,:,:s_mel_length[i]])
            tarmels.append(q_mel_padded[i:i+1][:,:,:q_mel_length[i]])
        return batches, refmels, tarmels





    #
