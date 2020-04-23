from torch import nn
import torch
import pdb


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

class EpisodicLoss(nn.Module):
    def __init__(self):
        super(EpisodicLoss, self).__init__()
        self.ma_et = 1.0
        self.ma_rate = 0.95

    def forward(self, model_output, targets):
        mel_target, gate_target, _ = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1,1)

        mel_out, mel_out_postnet, gate_out, _, t_et = model_output
        gate_out = gate_out.view(-1,1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        
        #t, et = t_et
        #self.ma_et = (1-self.ma_rate)*self.ma_et + self.ma_rate*torch.mean(et)
        #mi_loss = -(torch.mean(t) - (1/self.ma_et.mean()).detach()*torch.mean(et))

        #style_loss = nn.MSELoss()(style_embedding, style_target)
        return mel_loss + gate_loss #+ mi_loss
