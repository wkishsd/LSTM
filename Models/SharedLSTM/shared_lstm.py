"""One shared recurrent layer with explicit, caller-owned state."""

# Default network sizes. Edit independently before running the script.
INPUT_SIZE: int = 4   # Number of new input values supplied at each timestep.
HIDDEN_SIZE: int = 4  # Number of hidden values and cell values per stream.

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

LSTMState = Tuple[Tensor, Tensor]


def detach_state(state: LSTMState) -> LSTMState:
    """Return (hidden, cell) with gradient history removed, keeping their values.

    Args:
        state: The (hidden, cell) tensors returned by the encoder. Each has
            shape [1, batch, hidden_size].

    Returns:
        A pair of detached tensors sharing storage with the original tensors.
        This cuts gradient history without resetting the recurrent memory.
    """
    return state[0].detach(), state[1].detach()


class SharedLSTMEncoder(nn.Module):
    """Encode all input features together using one unidirectional LSTM layer.

    Args:
        input_size: Positive integer number of input values at each timestep.
        hidden_size: Positive integer number of hidden values and, separately,
            cell values per batch stream. Independent of input_size.

    Standard PyTorch initialization runs only at construction. There is no
    projection, dropout, fixed lookback, or internally retained sequence state.
    Memory can forget information. Units may learn different representations;
    distinct specialization is not guaranteed.
    """

    def __init__(
        self, input_size: int = INPUT_SIZE, hidden_size: int = HIDDEN_SIZE
    ) -> None:
        super().__init__()
        for name, value in (("input_size", input_size), ("hidden_size", hidden_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            proj_size=0,
            dropout=0.0,
        )

    def forward(
        self, x: Tensor, state: Optional[LSTMState] = None
    ) -> Tuple[Tensor, LSTMState]:
        """Process one timestep or a consecutive chunk, preserving gradients.

        Args:
            x: Floating-point tensor [batch, timesteps, input_size], with
                positive batch and timestep counts. Use the model's device
                and dtype (move the model with .to(...) as needed).
            state: Optional (hidden, cell) pair, both [1, batch, hidden_size]
                and on x's device with x's dtype. None starts from zeros.
                To continue a sequence, pass both tensors from its previous
                call and keep batch streams in the same order. No automatic
                detach is performed; use detach_state explicitly if needed.

        Returns:
            output: Hidden outputs at every timestep, shaped
                [batch, timesteps, hidden_size].
            (h, c): Final hidden and cell states, each [1, batch, hidden_size].
                The caller owns these states and decides whether to reuse them.

        Raises:
            TypeError: Configuration or tensors have invalid types or dtypes.
            ValueError: Shapes or devices are incompatible.
        """
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(f"x must have shape [batch, timesteps, {self.input_size}]")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("x must have positive batch and timestep counts")
        weight = self.lstm.weight_ih_l0
        if x.device != weight.device:
            raise ValueError("x and the model must be on the same device")
        if not x.is_floating_point() or x.dtype != weight.dtype:
            raise TypeError("x must have the same floating-point dtype as the model")
        if state is not None:
            if not isinstance(state, tuple) or len(state) != 2:
                raise TypeError("state must be a (hidden, cell) tuple")
            expected = (1, x.shape[0], self.hidden_size)
            for name, value in zip(("hidden", "cell"), state):
                if not isinstance(value, Tensor):
                    raise TypeError(f"{name} state must be a torch.Tensor")
                if tuple(value.shape) != expected:
                    raise ValueError(f"{name} state must have shape {expected}")
                if value.device != x.device:
                    raise ValueError(f"{name} state must be on x's device")
                if value.dtype != x.dtype:
                    raise TypeError(f"{name} state must have x's dtype")
        return self.lstm(x, state)
