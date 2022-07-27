import torch
import torch.nn as nn
import torch.nn.functional as F

def _log_transform(x):
    x = F.relu(x)
    return torch.log(1 + x)


def log_transform(x):
    yp = _log_transform(x)
    yn = _log_transform(-x)
    return yp - yn

def spectrogram(
        x, 
        n_fft=128, 
        hop_length=32, 
        window_fn=torch.hann_window, 
        log_transform=True, 
        magnitude=True,
        phase=False,
        grad=False,
        discard_zero_freq=True):
    """
    x: tensor of shape [batch, time_steps]
    n_fft: width of each FFT window (number of frequency bins is n_fft//2 + 1)
    hop_length: interval between consecutive windows
    window_fn: windowing function
    log_transform: if true, apply the function f(x) = log(1 + x) pointwise to the output of the spectrogram
    magnitude: if true, return the magnitude of the complex value in each time-frequency bin
    grad: if true, allow gradients to propagate through the spectrogram transformation
    discard_zero_freq: if true, remove the zero frequency row from the spectrogram
    """    
    with torch.set_grad_enabled(grad):
        window = window_fn(n_fft).to(x.device)
        stft = torch.stft(x, n_fft=n_fft, window=window, hop_length=hop_length, center=True)
        stft = stft[..., :x.shape[-1]//hop_length, :]
        if discard_zero_freq:
            stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3]-1)
        if magnitude:
            stft_mag = torch.norm(stft, dim=-1)
            if log_transform:
                stft_mag = torch.log(1 + F.relu(stft_mag))
            if phase:
                components = stft.split(1, dim=-1)
                stft_phase = torch.atan2(components[1].squeeze(-1), components[0].squeeze(-1))
                stft = torch.stack([stft_mag, stft_phase], dim=-1)
            else:
                stft = stft_mag
        else:
            if log_transform:
                stft = torch.log(1 + F.relu(stft)) - torch.log(1 + F.relu(-stft))
        return stft.clamp(-10, 10)


class FeatureExtractor(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64, 128), strides=(4, 2, 2, 2), bn=True,
                 pool=nn.MaxPool1d(2), kernel_sizes=[7,5,5,5], dropout=nn.Dropout2d(0.2),
                 nonlin=nn.LeakyReLU(0.1)):
        """Temporal convolutional network that extracts features for each station independently."""
        super().__init__()
        self.channels = channels
        self.strides = strides
        self.bn = bn
        self.pool = pool
        self.dropout = dropout
        self.kernel_sizes = kernel_sizes
        self.nonlin = nonlin
        # self.padding = nn.ReflectionPad1d(kernel_size // 2)
        self.paddings = nn.ModuleList([nn.ReflectionPad1d(k // 2) for k in kernel_sizes])
        if self.bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(c) for c in channels[1:]])
            conv_bias = False
        else:
            self.bn_layers = [lambda x: x for c in channels[1:]]
            conv_bias = True
        self.conv_layers = nn.ModuleList([
                             nn.Conv1d(channels[i], channels[i + 1], 
                                       kernel_sizes[i], stride=strides[i], 
                                       bias=conv_bias) 
                             for i in range(len(channels) - 1)])


    def forward(self, x):
        """input shape [batch, num_stations, in_channels, time_steps]
           output shape [batch, num_stations, out_channels, time_steps]"""
        x = log_transform(x).clamp(-10, 10)
        out = x.view(-1, *x.shape[-2:])
        for i, (pad, conv, bn) in enumerate(zip(self.paddings, self.conv_layers, self.bn_layers)):
            # out = self.padding(out)
            out = pad(out)
            out = bn(conv(out))
            # out = self.pool(out)
            # if i == 0:
            #     out = torch.tanh(out)
            # el
            if i < len(self.conv_layers) - 1:
                out = self.nonlin(out)
        out = out.view(*x.shape[:2], out.shape[1], -1)
        out = self.dropout(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, nonlin, stride=1, activate_before_residual=False, bn_momentum=1e-3):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.nonlin = nonlin
        self.activate_before_residual = activate_before_residual
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, padding_mode='reflect', bias=False)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(c_in, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm2d(c_out, momentum=bn_momentum)
        if c_in != c_out:
            self.conv_shortcut = nn.Conv2d(c_in, c_out, 1, stride=stride, padding=0, padding_mode='reflect', bias=False)

    def forward(self, x):
        out = self.nonlin(self.bn1(x))
        if self.activate_before_residual:
            x = out
        if self.c_in != self.c_out:
            x = self.conv_shortcut(x)
        out = self.conv1(out)
        out = self.nonlin(self.bn2(out))
        out = self.conv2(out)
        return x + out
    

# Transforms input into an amplitude spectrogram and extracts features using a WideResNet
# WideResNet architecture adapted from https://github.com/YU1ut/MixMatch-pytorch/blob/master/models/wideresnet.py
class ResNetFeatureExtractor(nn.Module):
    def __init__(
            self,
            output_dim=128,
            input_channels=3, 
            input_freq_dim=64,
            t_hop_length=32,
            filter_multiplier=32, 
            block_depth=1, 
            nonlin=nn.LeakyReLU(0.1), 
            bn_momentum=0.01):
        super().__init__()
        self.output_dim = output_dim
        self.input_channels = input_channels
        self.input_freq_dim = input_freq_dim
        self.t_hop_length = t_hop_length
        self.filter_multiplier = filter_multiplier
        self.block_depth = block_depth
        self.nonlin = nonlin

        nf = filter_multiplier
        self.conv = nn.Conv2d(input_channels, 16, 3, stride=1, padding=1, padding_mode='reflect', bias=False)
        self.block1 = nn.Sequential(
            ResidualBlock(16, nf, nonlin, stride=(2, 1), activate_before_residual=True, bn_momentum=bn_momentum),
            *[ResidualBlock(nf, nf, nonlin, bn_momentum=bn_momentum) for _ in range(block_depth - 1)])

        self.block2 = nn.Sequential(
            ResidualBlock(nf, nf*2, nonlin, stride=(2, 1), bn_momentum=bn_momentum),
            *[ResidualBlock(nf*2, nf*2, nonlin, bn_momentum=bn_momentum) for _ in range(block_depth - 1)])

        self.block3 = nn.Sequential(
            ResidualBlock(nf*2, nf*4, nonlin, stride=(2, 1), bn_momentum=bn_momentum),
            *[ResidualBlock(nf*4, nf*4, nonlin, bn_momentum=bn_momentum) for _ in range(block_depth - 1)])

        self.bn = nn.BatchNorm2d(nf*4, momentum=bn_momentum)
        self.fc = nn.Linear(nf * input_freq_dim // 2, output_dim)

    def forward(self, x, autocast=False):
        """input shape [batch, num_stations, in_channels, time_steps]
           output shape [batch, num_stations, out_channels, time_steps]"""
        with torch.cuda.amp.autocast(enabled=autocast):
            shape = x.shape
            sgram = spectrogram(x.view(-1, x.shape[-1]), n_fft=self.input_freq_dim*2, hop_length=self.t_hop_length)
            sgram = sgram.view(-1, self.input_channels, *sgram.shape[-2:])
            out = self.conv(sgram)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.nonlin(self.bn(out))
            out = out.view(out.shape[0], -1, out.shape[-1])
            out = self.fc(out.transpose(1, 2)).transpose(1, 2)
            out = out.view(x.shape[0], x.shape[1], *out.shape[-2:])
        return out


class Classifier(nn.Module):
    def __init__(self, channels=(128, 64), bn=True, nonlin=nn.LeakyReLU(0.1)):
        super().__init__()
        self.channels = channels
        self.bn = bn
        self.nonlin = nonlin

        if self.bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(c) for c in channels[1:]])
            conv_bias = False
        else:
            self.bn_layers = [lambda x: x for c in channels[1:]]
            conv_bias = True
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(channels[i], channels[i+1], 3, padding=1, padding_mode='reflect', bias=conv_bias) 
            for i in range(len(channels) - 1)])
        self.conv_out = nn.Conv1d(channels[-1], 1, kernel_size=3, padding=1, padding_mode='reflect')

    def forward(self, x):
        """input shape [batch, in_channels, time_steps]
           output shape [batch, time_steps]"""
        out = x
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            out = self.nonlin(bn(conv(out)))
        out = self.conv_out(out).squeeze(1)
        return out

class ClassifierWide(nn.Module):
    def __init__(self, channels=(128, 64, 32, 16), dilations=(2, 2, 2), bn=True, 
                 pool=nn.MaxPool1d(2), kernel_size=3, 
                 nonlin=nn.LeakyReLU(0.1)):
        super().__init__()
        self.channels = channels
        self.dilations = dilations
        self.bn = bn
        self.pool = pool
        self.kernel_size = kernel_size
        self.nonlin = nonlin
        self.paddings = nn.ModuleList([nn.ReflectionPad1d(((kernel_size-1)*d+1) // 2) for d in dilations])
        self.padding_out = nn.ReflectionPad1d(((kernel_size-1)*2+1) // 2)

        if self.bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(c) for c in channels[1:]])
            conv_bias = False
        else:
            self.bn_layers = [lambda x: x for c in channels[1:]]
            conv_bias = True
        self.conv_layers = nn.ModuleList([
                             nn.Conv1d(channels[i], channels[i + 1], 
                                       kernel_size, dilation=dilations[i],
                                       bias=conv_bias) 
                             for i in range(len(channels) - 1)])
        self.conv_out = nn.Conv1d(channels[-1], 1, kernel_size=kernel_size, dilation=2, bias=True)

    def forward(self, x):
        """input shape [batch, in_channels, time_steps]
           output shape [batch, time_steps]"""
        out = x
        for pad, conv, bn in zip(self.paddings, self.conv_layers, self.bn_layers):
            out = pad(out)
            out = self.nonlin(bn(conv(out)))
        out = self.padding_out(out)
        out = self.conv_out(out).squeeze(1)
        return out

class End2End(nn.Module):
    def __init__(self, spectrogram=True):
        super().__init__()
        if spectrogram:
            self.feature_extractor = ResNetFeatureExtractor()
            self.classifier = Classifier()
            # self.classifier = ClassifierWide()
        else:
            self.feature_extractor = FeatureExtractor()
            self.classifier = ClassifierWide()
    
    def forward(self, x, decays=None):
        features = self.feature_extractor(x)
        if decays is not None:
            decays = decays.unsqueeze(2).unsqueeze(3)
            features = (features * decays).mean(1)
        else:
            features = features.mean(1)
        logits = self.classifier(features)
        return logits