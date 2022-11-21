import argparse
import os

import torch
import yaml
import torch.nn as nn

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample, plot_spectrogram_to_numpy, plot_alignment_to_numpy
from model import FastSpeech2Loss
from data_utils import AudioTextDataset, AudioTextCollate, DataLoader
import matplotlib
MATPLOTLIB_FLAG = False

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = AudioTextDataset(
        preprocess_config['path']['validation_files'], preprocess_config)
    
    batch_size = train_config["optimizer"]["batch_size"]
    collate_fn = AudioTextCollate()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=8, 
        pin_memory=True, 
        drop_last=False
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config, train_config).to(device)

    # Evaluation
    loss_sums = [0, 0, 0, 0, 0, 0]
    for batch in loader:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(*(batch), step=step, gen=False)

            # Cal Loss
            losses = Loss(batch, output, step=step)
            
            for i in range(len(losses)):
                loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Pitch Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        loss_means = {
            "loss/total_loss": loss_means[0],
            "loss/mel_loss": loss_means[1],
            "loss/pitch_loss": loss_means[2],
            "loss/duration_loss": loss_means[3],
            "loss/ctc_loss": loss_means[4],
            "loss/bin_loss": loss_means[5],
        }
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        wav_reconstruction, wav_prediction = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )
        mels_orig = batch[4]
        specs = output[0]
        attn_h = output[8]
        attn_s = output[9]
        filters_0 = output[11]
        sources_0 = output[12]
        
        image_dict = {
            "spec/mel_orig": plot_spectrogram_to_numpy(mels_orig[0].detach().cpu().numpy()), 
            "spec/mel_0": plot_spectrogram_to_numpy(specs[0][0].detach().cpu().numpy()), 
            "spec/mel_1": plot_spectrogram_to_numpy(specs[1][0].detach().cpu().numpy()), 
            "spec/mel_2": plot_spectrogram_to_numpy(specs[2][0].detach().cpu().numpy()), 
            "spec/filter": plot_spectrogram_to_numpy(filters_0[0].detach().cpu().numpy()), 
            "spec/source": plot_spectrogram_to_numpy(sources_0[0].detach().cpu().numpy()), 
            "attn/attn_h": plot_alignment_to_numpy(attn_h[0,0].data.detach().cpu().numpy()), 
            "attn/attn_s": plot_alignment_to_numpy(attn_s[0,0].data.detach().cpu().numpy())
        }
        audio_dict = {
          "wav/audio": wav_reconstruction / 32768, 
          "wav/audio_gen": wav_prediction / 32768
        }
        if logger is not None:
            log(writer=logger,
                global_step=step, 
                images=image_dict,
                audios=audio_dict, 
                scalars=loss_means, 
                audio_sampling_rate=sampling_rate)

    return message
