import math

import torch
from torch import nn

from .schema import ENCODER_WIDTHS, FEATURE_INPUT_LABELS, MODEL_LABELS, STATES, WIDTH, window_input_size


PRICE_COLUMNS = slice(0, 36, 9)
VOLUME_COLUMNS = slice(1, 36, 9)
PRICE_RELATIVE_INPUTS = ("relative_price", "stationary")
MACD_RELATIVE_INPUTS = ("relative_macd", "stationary")
# Largest float32 below 1; keeps the inverse transform finite if a stored value rounds to +-1.
BOUNDED_LIMIT = 1 - 2.0 ** -24


def relative_macd_features(features):
    # The frozen timeline contains x / sqrt(1 + x^2) and ln(close).
    market = features[..., :36].unflatten(-1, (4, 9))
    bounded = market[..., 4:6].clamp(-BOUNDED_LIMIT, BOUNDED_LIMIT)
    unbounded = bounded / torch.sqrt((1 - bounded) * (1 + bounded))
    result = features.clone()
    result[..., :36].unflatten(-1, (4, 9))[..., 4:6] = unbounded * torch.exp(-market[..., :1])
    return result


class PerMinuteLinear(nn.Module):
    """An independent affine transform for each minute, batched across minutes."""

    def __init__(self, minutes, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(minutes, inputs, outputs))
        self.bias = nn.Parameter(torch.empty(minutes, 1, outputs))
        bound = 1 / math.sqrt(inputs)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.baddbmm(self.bias, x.transpose(0, 1), self.weight).transpose(0, 1)


class TradeMLP(nn.Module):
    def __init__(self, hidden1=128, hidden2=64, activation="relu", standardize=False,
                 dropout=0.0, lookback_minutes=STATES, hidden3=0, model="dense",
                 feature_input="exported", num_classes=2):
        super().__init__()
        if hidden1 < 1 or hidden2 < 1:
            raise ValueError("Hidden layer widths must be positive")
        if hidden3 < 0:
            raise ValueError("Third hidden layer width must be nonnegative (0 disables it)")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("Dropout must be a finite fraction from 0 up to but excluding 1")
        self.input_size = window_input_size(lookback_minutes)
        self.lookback_minutes = lookback_minutes
        self.num_classes = num_classes
        activations = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        if model not in MODEL_LABELS:
            raise ValueError(f"Unknown model: {model}")
        if feature_input not in FEATURE_INPUT_LABELS:
            raise ValueError(f"Unknown feature input: {feature_input}")
        self.feature_input = feature_input
        activate = activations[activation]
        self.encoder = None
        head_inputs = self.input_size
        if model != "dense":
            hidden, encoded = ENCODER_WIDTHS
            if model == "shared":
                first, second = nn.Linear(WIDTH, hidden), nn.Linear(hidden, encoded)
            else:
                first = PerMinuteLinear(lookback_minutes, WIDTH, hidden)
                second = PerMinuteLinear(lookback_minutes, hidden, encoded)
            self.encoder = nn.Sequential(first, activate(), second, activate())
            head_inputs = lookback_minutes * encoded

        widths = [head_inputs, hidden1, hidden2]
        if hidden3:
            widths.append(hidden3)
        layers = []
        for input_width, output_width in zip(widths, widths[1:]):
            layers.extend((nn.Linear(input_width, output_width), activate()))
        layers.append(nn.Linear(widths[-1], num_classes))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("feature_mean", torch.zeros(WIDTH) if standardize else None)
        self.register_buffer("feature_scale", torch.ones(WIDTH) if standardize else None)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"Expected (batch, {self.input_size}), got {tuple(x.shape)}")
        x = x.reshape(-1, self.lookback_minutes, WIDTH)
        if self.feature_input in MACD_RELATIVE_INPUTS:
            x = relative_macd_features(x)
        prices = x[..., PRICE_COLUMNS]
        if self.feature_mean is not None:
            x = (x - self.feature_mean) / self.feature_scale
        elif self.feature_input in PRICE_RELATIVE_INPUTS:
            x = x.clone()  # never modify the caller's batch in place
        if self.feature_input in PRICE_RELATIVE_INPUTS:
            # Log return in percent units, relative to the last closed candle for each timeframe.
            x[..., PRICE_COLUMNS] = 100 * (prices - prices[:, -1:, :])
        if self.feature_input == "stationary":
            # ln(volume) relative to its mean over this window, so the absolute level cannot identify the period.
            volume = x[..., VOLUME_COLUMNS]
            x[..., VOLUME_COLUMNS] = volume - volume.mean(dim=1, keepdim=True)
        if self.encoder is not None:
            x = self.encoder(x)
        x = x.flatten(1)
        for index in range(0, len(self.layers) - 1, 2):
            x = self.dropout(self.layers[index + 1](self.layers[index](x)))
        return self.layers[-1](x)

    @torch.inference_mode()
    def probabilities(self, x):
        return self(x).softmax(dim=1)
