from torch import nn
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

    def forward(self, model_output, targets):
        mel_target, gate_target, style_target = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1,1)

        mel_out, mel_out_postnet, gate_out, _, style_embedding = model_output
        gate_out = gate_out.view(-1,1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
            
        style_loss = nn.MSELoss()(style_embedding, style_target)
        return mel_loss + gate_loss + style_loss
