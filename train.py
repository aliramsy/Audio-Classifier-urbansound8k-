import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import AudioClassifier
from UrbanSoundDataset import UrbanSoundDataset
import torchaudio
import matplotlib.pyplot as plt

batch_size = 16
epochs = 10
lr = .01
audio_dir = './urban_sounds_small/urban_sounds_small'
anotation_file = './urban_sounds_small/urban_sounds_small/metadata.csv'
sample_rate = 44100
seconds = 10
num_samples = int(seconds * sample_rate)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
validation_ratio = 0.1

mel_spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                      n_mels=64,
                                                      n_fft=2048,
                                                      hop_length=512)

usd = UrbanSoundDataset(audio_dir, anotation_file,
                        mel_spectogram, sample_rate, num_samples, device)

total_size = len(usd)
validation_size = int(total_size * validation_ratio)
training_size = total_size - validation_size
train_dataset, val_dataset = random_split(
    usd, [training_size, validation_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

audioClassifier = AudioClassifier(input_channel=1, num_classes=9).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(audioClassifier.parameters(), lr=lr)

for epoch in range(epochs):
    print(f'in epoch number {epoch}:/n')
    audioClassifier.train()
    train_loss = 0
    train_acc = 0
    total_correct = 0
    total = 0

    for signals, labels in train_loader:
        outputs = audioClassifier(signals)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        total += labels.size(0)
        total_correct += correct

    train_acc = total_correct/total
    train_loss = train_loss/len(train_loader)

    # Validation
    audioClassifier.eval()
    val_loss = 0
    val_acc = 0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in val_loader:
            outputs = audioClassifier(signals)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            correct = (preds == labels).sum().item()
            total += labels.size(0)
            total_correct += correct

        val_acc = total_correct/total
        val_loss = train_loss/len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
