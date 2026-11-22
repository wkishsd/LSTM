"""Run with: python example_shared_lstm.py (requires PyTorch)."""

import torch

from shared_lstm import INPUT_SIZE, HIDDEN_SIZE, SharedLSTMEncoder, detach_state


def main() -> None:
    """Show two consecutive calls using already constructed synthetic tensors."""
    x_first = torch.arange(1, 2 * INPUT_SIZE + 1, dtype=torch.float32).reshape(2, 1, INPUT_SIZE) / 10
    x_next = x_first + 0.1
    encoder = SharedLSTMEncoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)

    output_first, (h, c) = encoder(x_first)  # Start from zeros.
    print("first:", output_first.shape, h.shape, c.shape)
    # Expected: [2, 1, HIDDEN_SIZE], [1, 2, HIDDEN_SIZE], [1, 2, HIDDEN_SIZE].

    output_next, (h, c) = encoder(x_next, (h, c))  # Continue both streams.
    print("next: ", output_next.shape, h.shape, c.shape)
    # Same shapes as above. Gradients span both calls.

    detached_h, detached_c = detach_state((h, c))  # Optional, explicit cutoff.
    assert not detached_h.requires_grad and not detached_c.requires_grad


if __name__ == "__main__":
    main()
