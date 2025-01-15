from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import torch


class UrbanSoundDataset(Dataset):
    def __init__(self, audio_dir, annotation_file, transformation, sample_rate, num_samples, device):
        self.num_samples = num_samples
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.transformation = transformation.to(device)
        self.sample_rate = sample_rate
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_file = self.get_audio_file(index)
        label = self.get_label(index)
        signal, sr = torchaudio.load(audio_file)
        signal = signal.to(self.device)

        signal = self.resample_signal(signal, sr, self.sample_rate)
        signal = self.donw_sample_signal(signal)

        if signal.shape[1] > self.num_samples:
            signal = self.cut_samples(signal)
        elif signal.shape[1] < self.num_samples:
            signal = self.right_pad(signal)

        signal = self.transformation(signal)

        return signal, label

    def resample_signal(self, signal, pre_sample_rate, sample_rate):
        if pre_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(
                pre_sample_rate, sample_rate)
            signal = resampler(signal)
        return signal

    def donw_sample_signal(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def get_audio_file(self, index):
        class_dir = self.annotations.iloc[index, 1]
        file_name = self.annotations.iloc[index, 0]
        audio_file = os.path.join(self.audio_dir, class_dir, file_name)
        return audio_file

    def cut_samples(self, signal):
        signal = signal[:, :self.num_samples]
        return signal

    def right_pad(self, signal):
        num_missing_samples = self.num_samples - signal.shape[1]
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def get_label(self, index):
        label = int(self.annotations.iloc[index, 1][:2])
        if label < 2:
            label = label - 1
        else:
            label = label - 2

        return torch.tensor(label, dtype=torch.long)


if __name__ == '__main__':
    audio_dir = './urban_sounds_small/urban_sounds_small'
    anotation_file = './urban_sounds_small/urban_sounds_small/metadata.csv'
    sample_rate = 44100
    seconds = 10
    num_samples = int(seconds * sample_rate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device is {device}')

    mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                          n_mels=64,
                                                          n_fft=2048,
                                                          hop_length=512)

    usd = UrbanSoundDataset(audio_dir, anotation_file,
                            mel_spectogram, sample_rate, num_samples, device)
    print(usd[0][0].shape)
