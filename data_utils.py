from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from scipy.stats import betabinom
from tqdm.auto import tqdm
import pyworld as pw
import numpy as np
import librosa
import random
import torch
import json
import os

from utils.tools import intersperse
from text import text_to_sequence
import audio as Audio
random.seed(0)



class AudioTextCollate(object):
    def __call__(self, batch):
        # speaker_id, text, mel, pitch, attn_prior
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(-1) for x in batch]),
            dim=0, descending=True)
        
        max_text_len = max([len(x[1]) for x in batch])
        max_spec_len = max([x[2].size(1) for x in batch])
        
        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        
        sid = torch.LongTensor(len(batch))
        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_len)
        pitch_padded = torch.FloatTensor(len(batch), max_spec_len)
        attn_prior_padded = torch.FloatTensor(len(batch), max_text_len, max_spec_len)
        
        sid.zero_()
        text_padded.zero_()
        spec_padded.zero_()
        pitch_padded.zero_()
        attn_prior_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            
            sid[i] = row[0]
            
            text = row[1]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
            
            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            
            pitch = row[3]
            pitch_padded[i, :pitch.size(0)] = pitch
            
            attn_prior = row[4]
            attn_prior_padded[i, :text.size(0), :spec.size(1)] = attn_prior
            
        return (
            sid, 
            text_padded, 
            text_lengths, 
            max_text_len, 
            spec_padded, 
            spec_lengths, 
            max_spec_len, 
            pitch_padded, 
            attn_prior_padded
        )


    
class AudioTextDataset(Dataset):
    def __init__(self, file_path, preprocess_config):
        super(Dataset, self).__init__()
        self.processor = AudioTextProcessor(preprocess_config, False)
        with open(file_path, 'r', encoding='utf8') as f:
            self.lines = f.read().split('\n')
            self.lines = [l for l in self.lines if len(l) > 0]
            tmp = []
            for line in self.lines:
                _, _, text = line.split('|')
                if len(text.strip()) > 1:
                    tmp.append(line)
            self.lines = tmp
            
    def get_values(self, line):
        values = self.processor(line)
        values[0] = torch.LongTensor([values[0]]).long()
        values[1] = torch.LongTensor(values[1]).long()
        values[2] = torch.from_numpy(values[2]).float()
        values[3] = torch.from_numpy(values[3]).float()
        values[4] = torch.from_numpy(values[4]).float()
        return values
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        return self.get_values(self.lines[index])

    
    
class AudioTextProcessor(object):
    def __init__(self, preprocess_config, preprocessing=False):
        self.preprocess_config = preprocess_config
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.filter_length = preprocess_config["preprocessing"]["stft"]["filter_length"]
        self.trim_top_db = preprocess_config["preprocessing"]["audio"]["trim_top_db"]
        self.trim_frame_length = preprocess_config["preprocessing"]["audio"]["trim_frame_length"]
        self.trim_hop_length = preprocess_config["preprocessing"]["audio"]["trim_hop_length"]
        self.beta_binomial_scaling_factor = preprocess_config["preprocessing"]["duration"]["beta_binomial_scaling_factor"]
        self.use_intersperse = preprocess_config["preprocessing"]["text"]["use_intersperse"]
        self.preprocessing = preprocessing
        
        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
        
        if not preprocessing:
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                self.pitch_mean, self.pitch_std = stats["pitch"][2:]
            
    def normalize(self, values, mean, std):
        return (values - mean) / std
        
    def load_audio(self, wav_path):
        wav_raw, _ = librosa.load(wav_path, self.sampling_rate)
        _, index = librosa.effects.trim(
            wav_raw, top_db=self.trim_top_db, 
            frame_length=self.filter_length, 
            hop_length=self.hop_length)
        wav_raw = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / self.hop_length
        return wav_raw.astype(np.float32), int(duration)
    
    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]
    
    def beta_binomial_prior_distribution(self, phoneme_count, mel_count, scaling_factor=1.0):
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M+1):
            a, b = scaling_factor*i, scaling_factor*(M+1-i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)
        
    def process_utterance(self, line):
        wav_path, speaker_id, text = line.split('|')
        text = text_to_sequence(text)
        if not self.preprocessing and self.use_intersperse:
            text = intersperse(text, 0)
        
        speaker_id = int(speaker_id)
        wav, duration = self.load_audio(wav_path)
        
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = self.remove_outlier(pitch[: duration])
        
        if self.preprocessing:
            if np.sum(pitch != 0) <= 1:
                return None
    
        mel_spectrogram = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : duration]
        
        attn_prior = self.beta_binomial_prior_distribution(
            mel_spectrogram.shape[1],
            len(text),
            self.beta_binomial_scaling_factor,
        )
        
        if not self.preprocessing:
            pitch = self.normalize(pitch, self.pitch_mean, self.pitch_std)
        
        return (
            speaker_id, 
            text, 
            mel_spectrogram, 
            pitch, 
            attn_prior
        )
    
    def __call__(self, line):
        return list(self.process_utterance(line))



class StatParser(AudioTextProcessor):
    def __init__(self, preprocess_config, preprocessing=True):
        super(StatParser, self).__init__(preprocess_config, preprocessing)
        self.corpus_path = preprocess_config['path']['corpus_path']
        self.row_path = preprocess_config['path']['raw_path']
        self.out_dir = preprocess_config['path']['preprocessed_path']
        self.val_size = preprocess_config['preprocessing']['val_size']
        self.pitch_normalization = preprocess_config['preprocessing']['pitch']['normalization']
        self.pitch_scaler = StandardScaler()
        with open(self.row_path, 'r', encoding='utf8') as f:
            self.lines = f.read().split('\n')
            self.lines = [l for l in self.lines if len(l) > 0]
            self.tmp = []
            self.pitches = []
            
    def normalize(self, values, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for value in values:
            value = (value - mean) / std
            max_value = max(max_value, max(value))
            min_value = min(min_value, min(value))
        return min_value, max_value
        
    def __call__(self):
        for line in tqdm(self.lines, total=len(self.lines)):
            line = line.split('|')
            line[0] = os.path.join(self.corpus_path, line[0])
            tmp = [line[0], str(0), line[3]]
            line = '|'.join(tmp)
            
            try:
                values = self.process_utterance(line)
            except:
                print(line)
                
            if values is None:
                continue
            _, _, _, pitch, _ = values
            if len(pitch) < 1:
                continue
                
            self.pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            self.tmp.append(line)
            self.pitches.append(pitch)
            
        if self.pitch_normalization:
            pitch_mean = self.pitch_scaler.mean_[0]
            pitch_std = self.pitch_scaler.scale_[0]
        else:
            pitch_mean = 0
            pitch_std = 1
            
        pitch_min, pitch_max = self.normalize(
            self.pitches, pitch_mean, pitch_std)
            
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
            }
            f.write(json.dumps(stats))
    
        with open(os.path.join(self.out_dir, 'train.txt'), 'w', encoding='utf8') as f:
            trn_lines = self.tmp[:-self.val_size]
            for line in trn_lines:
                f.write(f"{line}\n")
        with open(os.path.join(self.out_dir, 'val.txt'), 'w', encoding='utf8') as f:
            val_lines = self.tmp[-self.val_size:]
            for line in val_lines:
                f.write(f"{line}\n")
