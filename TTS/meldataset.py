# coding: utf-8
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from librosa.filters import mel as librosa_mel_fn
from pathlib import Path
from text_utils import TextCleaner
from torch.utils.data.sampler import WeightedRandomSampler
from librosa.util import normalize

import logging

MAX_WAV_VALUE = 32768.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_list,
        sr=22050,
        data_augmentation=False,
        validation=False,
    ):

        _data_list = [l[:-1].split("|") for l in data_list]
        self.min_seq_len = int(0.8 * 22050)
        self.text_cleaner = TextCleaner()
        self.sr = sr
        self.max_mel_length = 192

        self.data_list = self._filter(_data_list)

    def _filter(self, data):
        data_list = [
            (data[0], data[4], data[1], data[3])
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

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]

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

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, : (length_feature - length_feature % 2)]

        return speaker_id, acoustic_feature, text_tensor, path

    def _load_tensor(self, data):
        wave_path, text, speaker_id, _ = data
        wave, sr = sf.read(wave_path)
        audio = wave / MAX_WAV_VALUE
        wave = normalize(audio) * 0.95

        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 22050:
            wave = librosa.resample(wave, sr, 22050)
            print(wave_path, sr)

        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

        text = self.text_cleaner(text)

        text.insert(0, 0)
        text.append(0)

        text = torch.LongTensor(text)

        return wave, text, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
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

        for bid, (label, mel, text, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path

        if self.return_wave:
            return paths, texts, input_lengths, mels, output_lengths

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

    dataset = FilePathDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)

    if not validation:
        sampler = get_weighted_sampler(
            dataset.data_list,
            by_emotion=False,
            by_speaker=True,
            by_language=True,
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
