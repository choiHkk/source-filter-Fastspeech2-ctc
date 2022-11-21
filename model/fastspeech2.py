import os
import json
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, SourceGenerator, FilterGenerator, Decoder
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths



class FastSpeech2(nn.Module):
    """ FastSpeech2 with alignment learning """

    def __init__(self, preprocess_config, model_config, train_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.source_generator = SourceGenerator(model_config)
        self.filter_generator = FilterGenerator(model_config)
        self.decoder = Decoder(model_config, preprocess_config["preprocessing"]["mel"]["n_mel_channels"])

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None, 
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        attn_priors=None, 
        p_control=1.0,
        d_control=1.0,
        step=None, 
        gen=False, 
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            g = self.speaker_emb(speakers)
            output = output + g.unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            acoustics,
            linguistics, 
            p_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks, 
            attn_h, 
            attn_s, 
            attn_logprob
        ) = self.variance_adaptor(
            output, 
            src_lens, 
            src_masks,
            mels, 
            mel_lens, 
            mel_masks, 
            max_mel_len, 
            p_targets,
            attn_priors, 
            g, 
            p_control,
            d_control,
            step, 
            gen, 
        )

        specs, filters_0, sources_0, mel_masks = self.decoder(acoustics, linguistics, mel_masks)

        return (
            specs,
            p_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_h, 
            attn_s, 
            attn_logprob, 
            filters_0, 
            sources_0
        )
    