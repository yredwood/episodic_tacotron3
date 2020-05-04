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
    def __init__(self, hparams):
        super(EpisodicLoss, self).__init__()
        self.lin_freq = int(3000 / (hparams.sampling_rate * 0.5) * (hparams.n_lin_channels))

    def forward(self, pred, label):
        #mel_target, gate_target, style_target = targets
        label['mel'].requires_grad = False
        label['gate'].requires_grad = False
        label['gate'] = label['gate'].view(-1,1)

        pred['gate'] = pred['gate'].view(-1,1)

        mel_loss = nn.MSELoss()(pred['mel'], label['mel']) \
                + nn.MSELoss()(pred['mel_post'], label['mel'])

        lin_loss = 0.5 * nn.MSELoss()(pred['lin_post'], label['lin']) \
                + 0.5 * nn.MSELoss()(pred['lin_post'][:,:self.lin_freq] , label['lin'][:,:self.lin_freq])

        gate_loss = nn.BCEWithLogitsLoss()(pred['gate'], label['gate'])
        
        return mel_loss + gate_loss + lin_loss
