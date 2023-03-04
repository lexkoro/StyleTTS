# coding: utf-8

import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from librosa.filters import mel as librosa_mel_fn

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from text_utils import TextCleaner

np.random.seed(1)
random.seed(1)
DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), "word_index_dict_new.txt")


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    # if torch.min(y) < -1.0:
    #     print("min value is ", torch.min(y))
    # if torch.max(y) > 1.0:
    #     print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


# to_mel = torchaudio.transforms.MelSpectrogram(
#     sample_rate=22050, n_mels=80, n_fft=2048, win_length=1024, hop_length=256
# )
# mean, std = -4, 4


# def preprocess(wave):
#     mel_tensor = to_mel(wave)
#     mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
#     return mel_tensor


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, dict_path=DEFAULT_DICT_PATH, sr=22050):

        _data_list = [l[:-1].split("|") for l in data_list]
        self.min_seq_len = int(1.5 * 22050)
        self.data_list = [
            (data[0], data[4], data[1])
            for data in _data_list
            if ((Path(data[0]).stat().st_size // 2) > self.min_seq_len)
        ]
        np.random.shuffle(self.data_list)
        self.text_cleaner = TextCleaner(dict_path)
        self.sr = sr

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        wave, text_tensor, speaker_id = self._load_tensor(data)
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = mel_spectrogram(
            wave_tensor.unsqueeze(0), 2048, 80, 22050, 256, 1024, 0, None, False
        ).squeeze(0)

        if (text_tensor.size(0) + 1) >= (mel_tensor.size(1) // 3):
            mel_tensor = F.interpolate(
                mel_tensor.unsqueeze(0),
                size=(text_tensor.size(0) + 1) * 3,
                align_corners=False,
                mode="linear",
            ).squeeze(0)

        length_feature = mel_tensor.size(1)
        acoustic_feature = mel_tensor[:, : (length_feature - length_feature % 2)]

        return wave_tensor, acoustic_feature, text_tensor, data[0]

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        wave, sr = sf.read(wave_path)

        # phonemize the text
        text = self.text_cleaner(text)
        blank_index = self.text_cleaner.word_index_dictionary[" "]
        text.insert(0, blank_index)  # add a blank at the beginning (silence)
        text.append(blank_index)  # add a blank at the end (silence)

        text = torch.LongTensor(text)

        return wave, text, speaker_id


class Collater(object):
    """
    Args:
      return_wave (bool): if true, will return the wave data along with spectrogram.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        paths = ["" for _ in range(batch_size)]
        for bid, (_, mel, text, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            assert text_size < (mel_size // 2)

        if self.return_wave:
            waves = [b[0] for b in batch]
            return texts, input_lengths, mels, output_lengths, paths, waves

        return texts, input_lengths, mels, output_lengths


def build_dataloader(
    path_list,
    validation=False,
    batch_size=4,
    num_workers=1,
    device="cpu",
    collate_config={},
    dataset_config={},
):

    dataset = MelDataset(path_list, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return data_loader
