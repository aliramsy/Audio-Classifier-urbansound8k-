from torch import nn
from torchsummary import summary


class AudioClassifier(nn.Module):
    def __init__(self, input_channel, num_classes=9):
        super().__init__()

        self.channels = [input_channel, input_channel * 16,
                         input_channel * 32, input_channel * 64, input_channel * 128]

        self.conv = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(self.channels[i],
                          self.channels[i+1],
                          kernel_size=3, padding=2,
                          stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)) for i in range(len(self.channels) - 1)])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_channel * 128 * 5 * 55, 9)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        logits = self.linear(x)
        preds = self.softmax(logits)
        return preds


if __name__ == '__main__':
    audioClassifier = AudioClassifier(1, 9)
    summary(audioClassifier, (1, 64, 862))
