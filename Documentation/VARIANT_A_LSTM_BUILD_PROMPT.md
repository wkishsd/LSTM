# Build prompt: standalone variant A LSTM

Implement a small, reusable **shared LSTM module** in Python using PyTorch.
This task is only the LSTM itself. Do not implement data processing, a lookback
window, an MLP, feature selection, exchange integration, targets or a training
pipeline.

## Architecture and configurable sizes

Use one standard unidirectional, single-layer `torch.nn.LSTM` with
`batch_first=True`, no projection and no dropout.

Expose two independent constructor parameters:

```python
input_size: int = 4
hidden_size: int = 4
```

- `input_size` is the number of values supplied at each timestep.
- `hidden_size` is the number of shared memory units.
- With the defaults, each timestep supplies four input values, and the LSTM
  maintains **four hidden values and four cell values** per batch stream.
- This means eight recurrent state values, not four stacked layers and not
  four separate LSTMs for four inputs.
- All input features enter the same LSTM; its memory can combine them.
- Derive all dimensions from configuration. Changing input size must not
  automatically change memory size or vice versa.

## Public interface

Create a clearly named module, for example `SharedLSTMEncoder`.

Its forward method accepts:

```python
forward(x, state=None) -> (output, (h, c))
```

Shapes:

- `x`: `[batch, timesteps, input_size]`.
- `output`: `[batch, timesteps, hidden_size]`, the hidden output at each step.
- `h`: `[1, batch, hidden_size]`, the final hidden state.
- `c`: `[1, batch, hidden_size]`, the final cell state.

For one timestep with the defaults, `x` is `[batch, 1, 4]` and each final
state is `[1, batch, 4]`.

## State continuity

The caller owns the state explicitly:

1. For a new independent sequence, `state=None` initializes zero states.
2. The module returns the updated hidden and cell states.
3. The caller supplies both returned states with the next input.

Do not silently reset state between calls when continuing a sequence. Do not
retain hidden global state that could mix unrelated sequences.

There is **no fixed LSTM lookback period**. A call can process one timestep or a
chunk of consecutive timesteps; carrying its returned states into the next call
continues the same sequence. Finite memory can still forget information.

Preserve gradients through provided states by default. Provide a small,
explicit `detach_state(state)` helper for callers who need to cut gradient
history while keeping the state values. Do not detach states automatically.

Use standard PyTorch initialization once when constructing the module. Do not
clone one unit's parameters across all units or reinitialize weights on each
forward call. Different units can learn different representations, but do not
promise that they will specialize differently.

## Implementation and delivery

- Validate positive configuration sizes and compatible input/state shapes.
- Use compatible devices and dtypes for inputs and state.
- Keep the module compatible with normal `.train()`, `.eval()`, `.to(...)`
  and `state_dict()` usage.
- Let callers control gradient tracking; do not hard-code `no_grad()` inside
  the forward method.
- Deliver one model module plus a short usage example using already constructed
  synthetic tensors. Show the first call, a continuing call using returned
  `(h, c)`, and their expected shapes.
- Document every argument and returned value in plain English.
- Keep existing Bitfinex scripts and trading calculations unchanged.

Reference:
https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
