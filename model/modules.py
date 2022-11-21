import os
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from utils.tools import get_mask_from_lengths, b_mas, pad
from transformer import LinearNorm



class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config, train_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.aligner = AlignmentEncoder(preprocess_config, model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        
        self.binarization_start_steps = train_config["duration"]["binarization_start_steps"]
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.d_model = model_config["transformer"]["encoder_hidden"]
        assert self.pitch_feature_level == "frame_level"

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.proj = LinearNorm(
            model_config["transformer"]["encoder_hidden"], 
            model_config["transformer"]["encoder_hidden"]*2, 
            bias=True
        )
        
    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out).to(attn.device)

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def forward(
        self,
        linguistics,
        src_len, 
        src_mask,
        mel=None, 
        mel_len=None, 
        mel_mask=None, 
        max_mel_len=None, 
        pitch_target=None,
        attn_prior=None, 
        g=None, 
        p_control=1.0,
        d_control=1.0,
        step=None, 
        gen=False, 
    ):
        log_duration_prediction = self.duration_predictor(linguistics, src_mask)
        
        if not gen:
            attn_s, attn_logprob = self.aligner(
                queries=mel, keys=linguistics, mask=src_mask, attn_prior=attn_prior, g=g)
            attn_h = self.binarize_attention_parallel(attn_s, src_len, mel_len).detach()
            duration_rounded = attn_h.sum(2)[:, 0, :]
            if step < self.binarization_start_steps:
                linguistics = torch.bmm(attn_s.squeeze(1), linguistics)
            else:
                linguistics, mel_len = self.length_regulator(linguistics, duration_rounded, max_mel_len)
        else:
            attn_h, attn_s, attn_logprob = None, None, None
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            linguistics, mel_len = self.length_regulator(linguistics, duration_rounded, max_mel_len)
            mel_mask = get_mask_from_lengths(mel_len)
        
        linguistics = self.proj(linguistics)
        linguistics, pitch_features = torch.split(linguistics, self.d_model, dim=-1)
        
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            pitch_features, pitch_target, mel_mask, p_control
        )
        pitch_embedding = pitch_embedding + g.unsqueeze(1)

        return (
            pitch_embedding, 
            linguistics, 
            pitch_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask, 
            attn_h, 
            attn_s, 
            attn_logprob
        )
    
    
class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel, 
                            padding=(self.kernel -1) // 2,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out
    
    
class AlignmentEncoder(torch.nn.Module):
    """ Alignment Encoder for Unsupervised Duration Modeling """
    """From comprehensive transformer tts"""

    def __init__(self, preprocess_config, model_config):
        super(AlignmentEncoder, self).__init__()
        n_spec_channels = preprocess_config['preprocessing']['mel']['n_mel_channels']
        n_att_channels = model_config['variance_predictor']['filter_size']
        n_text_channels = model_config['transformer']['encoder_hidden']
        temperature = model_config['temperature']
        multi_speaker = model_config['multi_speaker']
        
        
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            Conv(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3, 
                padding=int((3 - 1) / 2), 
                bias=True,
                w_init='relu'
            ),
            nn.ReLU(),
            Conv(
                n_text_channels * 2,
                n_att_channels,
                kernel_size=1, 
                bias=True,
            ),
        )

        self.query_proj = nn.Sequential(
            Conv(
                n_spec_channels,
                n_spec_channels * 2,
                kernel_size=3, 
                padding=int((3 - 1) / 2), 
                bias=True,
                w_init='relu',
            ),
            nn.ReLU(),
            Conv(
                n_spec_channels * 2,
                n_spec_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(),
            Conv(
                n_spec_channels,
                n_att_channels,
                kernel_size=1, 
                bias=True,
            ),
        )

        if multi_speaker:
            self.key_spk_proj = nn.Linear(n_text_channels, n_text_channels)
            self.query_spk_proj = nn.Linear(n_text_channels, n_spec_channels)

    def forward(self, queries, keys, mask=None, attn_prior=None, g=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
            speaker_embed (torch.tensor): B x C tnesor of speaker embedding for multi-speaker scheme.
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        if g is not None:
            keys = keys + self.key_spk_proj(g.unsqueeze(1).expand(
                -1, keys.shape[1], -1
            ))
            queries = queries + self.query_spk_proj(g.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        keys_enc = self.key_proj(keys).transpose(1, 2)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries.transpose(1, 2)).transpose(1, 2)

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            # print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(attn_prior.transpose(1,2)[:, None] + 1e-8)
            #print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.unsqueeze(2).permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init)
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
