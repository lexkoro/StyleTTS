# coding: utf-8
"""
TODO:
- make TestDataset
- separate transforms
"""

import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from librosa.util import normalize
from torch.utils.data import DataLoader
from librosa.filters import mel as librosa_mel_fn
from audiomentations import (
    AddGaussianSNR,
    Compose,
    Gain,
    SevenBandParametricEQ,
    PolarityInversion,
)
import pyworld as pw
from pathlib import Path
from torchaudio import functional
from torch.utils.data.sampler import WeightedRandomSampler
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

MAX_WAV_VALUE = 32768.0


def load_wav(full_path, sr_target):
    sampling_rate, data = sf.read(full_path)
    if sampling_rate != sr_target:
        raise RuntimeError(
            "Sampling rate of the file {} is {} Hz, but the model requires {} Hz".format(
                full_path, sampling_rate, sr_target
            )
        )
    return data, sampling_rate


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


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_list,
        sr=22050,
        data_augmentation=True,
        validation=False,
        verbose=True,
    ):

        _data_list = [l[:-1].split("|") for l in data_list]
        self.min_seq_len = int(1.1 * 22050)
        self.sr = sr
        self.n_fft = 1024
        self.num_mels = 100
        self.hop_size = 256
        self.win_size = 1024
        self.fmin = 0
        self.fmax = None

        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192

        self.verbose = verbose

        # for silence detection
        self.zero_value = -10  # what the zero value is
        self.bad_F0 = 5  # if less than 5 frames are non-zero, it's a bad F0, try another algorithm
        self.augmentor = Compose(
            [
                AddGaussianSNR(min_snr_in_db=32, max_snr_in_db=64, p=0.3),
            ]
        )
        self.data_list = self._filter(_data_list)

    def _filter(self, data):
        data_list = [
            (data[0], data[4], data[1])
            for data in data
            if (
                (Path(data[0]).stat().st_size // 2) > self.min_seq_len
                and len(data[4]) > 5
            )
        ]
        print("data_list length: ", len(data))
        print("filtered data_list length: ", len(data_list))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def path_to_mel_and_label(self, path):
        wave_tensor = self._load_tensor(path)

        # use pyworld to get F0
        output_file = path + "_f0.npy"
        # check if the file exists
        if os.path.isfile(output_file):  # if exists, load it directly
            f0 = np.load(output_file)
        else:  # if not exist, create F0 file
            x = wave_tensor.numpy().astype("double")
            frame_period = self.hop_size * 1000 / self.sr
            _f0, t = pw.harvest(
                x, self.sr, f0_floor=50, f0_ceil=1000, frame_period=frame_period
            )
            if sum(_f0 != 0) < self.bad_F0:  # this happens when the algorithm fails
                _f0, t = pw.dio(
                    x, self.sr, f0_floor=50, f0_ceil=1000, frame_period=frame_period
                )  # if harvest fails, try dio
            f0 = pw.stonemask(x, _f0, t, self.sr)
            # save the f0 info for later use
            np.save(output_file, f0)

        f0 = torch.from_numpy(f0).float()

        if self.data_augmentation:
            augmented_audio = wave_tensor.unsqueeze(0).detach().numpy()
            augmented_audio = self.augmentor(augmented_audio, sample_rate=22050)
            augmented_audio = torch.FloatTensor(augmented_audio)

        mel_tensor = mel_spectrogram(
            augmented_audio if self.data_augmentation else wave_tensor.unsqueeze(0),
            2048,
            80,
            22050,
            256,
            1024,
            0,
            None,
            False,
        ).squeeze(0)

        mel_length = mel_tensor.size(1)

        f0 = f0[:mel_length]
        f0_zero = f0 == 0

        #######################################
        # You may want your own silence labels here
        # The more accurate the label, the better the resultss
        is_silence = torch.zeros(f0.shape)
        is_silence[f0_zero] = 1
        #######################################

        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[
                :, random_start : random_start + self.max_mel_length
            ]
            f0 = f0[random_start : random_start + self.max_mel_length]
            is_silence = is_silence[random_start : random_start + self.max_mel_length]

        if torch.any(torch.isnan(f0)):  # failed
            f0[torch.isnan(f0)] = self.zero_value  # replace nan value with 0

        return mel_tensor, f0, is_silence

    def __getitem__(self, idx):
        data, _, _ = self.data_list[idx]
        mel_tensor, f0, is_silence = self.path_to_mel_and_label(data)
        return mel_tensor, f0, is_silence

    def _load_tensor(self, data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        f0s = torch.zeros((batch_size, self.max_mel_length)).float()
        is_silences = torch.zeros((batch_size, self.max_mel_length)).float()

        for bid, (mel, f0, is_silence) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            f0s[bid, :mel_size] = f0
            is_silences[bid, :mel_size] = is_silence

        if self.max_mel_length > self.min_mel_length:
            random_slice = (
                np.random.randint(
                    self.min_mel_length // self.mel_length_step,
                    1 + self.max_mel_length // self.mel_length_step,
                )
                * self.mel_length_step
                + self.min_mel_length
            )
            mels = mels[:, :, :random_slice]
            f0 = f0[:, :random_slice]

        mels = mels.unsqueeze(1)
        return mels, f0s, is_silences


def build_dataloader(
    path_list,
    validation=False,
    batch_size=4,
    num_workers=1,
    device="cpu",
    collate_config={},
    dataset_config={},
):

    dataset = MelDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)

    if not validation:
        sampler = get_weighted_sampler(
            dataset.data_list,
            by_emotion=False,
            by_speaker=True,
            by_language=False,
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=(not validation),
        sampler=sampler if not validation else None,
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return data_loader


def get_weighted_sampler(items, by_speaker=False, by_language=False, by_emotion=False):

    dataset_samples_weight = 1.0

    # language
    if by_language:
        language_names = np.array([item[3] for item in items])
        unique_language_names = np.unique(language_names).tolist()
        language_ids = [unique_language_names.index(l) for l in language_names]
        language_count = np.array(
            [len(np.where(language_names == l)[0]) for l in unique_language_names]
        )
        weight_language = 1.0 / language_count

        language_samples_weight = np.array(
            np.array([weight_language[l] for l in language_ids])
        )
        language_samples_weight = language_samples_weight / np.linalg.norm(
            language_samples_weight
        )

        language_samples_weight = torch.from_numpy(language_samples_weight).float()
        print("language_samples_weight", language_samples_weight)
        dataset_samples_weight += language_samples_weight * 1.5

    # speaker
    if by_speaker:
        speaker_names = np.array([item[2] for item in items])
        unique_speaker_names = np.unique(speaker_names).tolist()
        speaker_ids = [unique_speaker_names.index(l) for l in speaker_names]
        speaker_count = np.array(
            [len(np.where(speaker_names == l)[0]) for l in unique_speaker_names]
        )
        weight_speaker = 1.0 / speaker_count

        speaker_samples_weight = np.array(
            np.array([weight_speaker[l] for l in speaker_ids])
        )
        speaker_samples_weight = speaker_samples_weight / np.linalg.norm(
            speaker_samples_weight
        )
        speaker_samples_weight = torch.from_numpy(speaker_samples_weight).float()
        print("speaker_samples_weight", speaker_samples_weight)
        dataset_samples_weight += speaker_samples_weight * 1.2

    # emotion
    if by_emotion:
        emotion_names = np.array([item[2] for item in items])
        unique_emotion_names = np.unique(emotion_names).tolist()
        emotion_ids = [unique_emotion_names.index(l) for l in emotion_names]
        emotion_count = np.array(
            [len(np.where(emotion_names == l)[0]) for l in unique_emotion_names]
        )
        weight_emotion = 1.0 / emotion_count

        print(weight_emotion * 10000)

        emotion_samples_weight = np.array(
            np.array([weight_emotion[l] for l in emotion_ids])
        )
        emotion_samples_weight = emotion_samples_weight / np.linalg.norm(
            emotion_samples_weight
        )
        emotion_samples_weight = torch.from_numpy(emotion_samples_weight).float()
        print("emotion_samples_weight", emotion_samples_weight)
        dataset_samples_weight += emotion_samples_weight

    # dataset_samples_weight = (speaker_samples_weight * 1.5) + (emotion_samples_weight)
    # dataset_samples_weight = (language_samples_weight * 2) + (speaker_samples_weight)
    return WeightedRandomSampler(dataset_samples_weight, len(dataset_samples_weight))
