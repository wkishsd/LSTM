# Shared LSTM encoder — standalone variant A

The implementation is `shared_lstm.py`. It contains one standard PyTorch LSTM
layer. All features enter that same layer. The number of input values and the
number of memory units are independent settings, both defaulting to 4.

Edit `INPUT_SIZE` and `HIDDEN_SIZE` at the top of `shared_lstm.py` before
running to change the defaults. The example uses these settings and constructs
matching synthetic inputs. You can still override either size when constructing
`SharedLSTMEncoder(input_size=..., hidden_size=...)`.

## Architecture preview

```mermaid
flowchart LR
    X["One new observation<br/>4 input values per stream"] --> L["One shared LSTM layer<br/>4 memory units combine all inputs"]
    S["Memory from the previous call<br/>4 hidden values + 4 cell values<br/>Start with zeros for a new sequence"] --> L
    L --> Y["Current output<br/>4 updated hidden values"]
    L --> N["Updated memory<br/>4 hidden values + 4 cell values"]
    N --> NEXT["Caller saves both sets of values<br/>and supplies them with the next observation"]
```

This diagram uses the default sizes and shows one independent stream. A batch
processes multiple independent streams together, each with its own memory.
Each stream has eight recurrent state values inside one layer. The hidden
values are also the current output; they are not a third set of memory values.
Cell values carry the internal memory that the gates update.

**One new observation per call; no lookback window.** Previous input rows are
not collected or replayed. Only the hidden and cell values are carried forward.
The same learned layer processes each new observation, and finite memory can
still forget information.

## Operations inside the shared layer

The integrated left-to-right diagram follows every connection from incoming
values through the gates to updated cell memory and hidden output. For clarity,
the illustration uses two inputs and two memory units; the Python defaults
remain four inputs and four memory units. Both illustrated units belong to the
same shared layer. Plain-English step annotations appear below the diagram.

![Complete shared LSTM process with two inputs and two memory units](../../Documentation/shared_lstm_integrated.png)

Every box below uses descriptive names. Each gate produces four control values
with the default memory size, one for each memory unit.

```mermaid
flowchart LR
    X["New observation<br/>All 4 input values"] --> A["Four learned calculations<br/>Weighted sums of all new inputs<br/>and all previous hidden values, plus biases"]
    H["Previous hidden values<br/>4 values from the last call"] --> A
    A --> F["Forget gate: how much old memory to keep<br/>Sigmoid gives controls between 0 and 1"]
    A --> I["Input gate: how much new content to store<br/>Sigmoid gives controls between 0 and 1"]
    A --> G["Candidate memory: proposed new content<br/>Tanh gives values between -1 and 1"]
    A --> O["Output gate: how much memory to expose<br/>Sigmoid gives controls between 0 and 1"]
    C0["Previous cell memory<br/>4 values from the last call"] --> M1["Retained old memory<br/>Multiply old cell values by forget controls"]
    F --> M1
    I --> M2["Selected new content<br/>Multiply candidate values by input controls"]
    G --> M2
    M1 --> C["Updated cell memory<br/>Add retained old memory and selected new content"]
    M2 --> C
    C --> T["Apply tanh to updated cell memory<br/>Result lies between -1 and 1"]
    T --> HT["Updated hidden values<br/>Multiply bounded memory by output controls"]
    O --> HT
    HT --> OUT["Return as current output<br/>Also carry forward as hidden memory"]
    C --> CS["Carry forward as cell memory<br/>The cell memory itself is not bounded by tanh"]
```

The gates are calculations within the one LSTM layer. Each has its own learned
weights and biases. Every memory unit can receive information from every input
feature and every previous hidden value.

The mathematical operations, written with descriptive names, are:

- **Each learned calculation** = input weights × new observation + hidden
  weights × previous hidden values + input bias + hidden bias.
- **Updated cell memory** = forget controls × previous cell memory + input
  controls × candidate memory.
- **Updated hidden values** = output controls × tanh(updated cell memory).

The first equation uses matrix multiplication to mix inputs. The last two
multiply corresponding values one by one, then add corresponding values.
Sigmoid smoothly maps a number into the range 0 to 1. Tanh smoothly maps a
number into the range -1 to 1. These controls are learned, continuous amounts,
not fixed on/off switches.

Source: [PyTorch LSTM reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html).

## Usage

`example_shared_lstm.py` constructs synthetic tensors and shows a first call,
a continuing call, and optional explicit state detachment. Each call has
input/output shape [2, 1, 4]; both final states have shape [1, 2, 4].

Run using the project-local test environment:

```powershell
.\.venv-shared-lstm\Scripts\python.exe Models/SharedLSTM/example_shared_lstm.py
.\.venv-shared-lstm\Scripts\python.exe -m unittest discover -s Models/SharedLSTM -v
```

For one-observation-at-a-time use, supply input shape
`[number of streams, 1, number of input values]`. The middle dimension of 1
means one new observation, not a lookback window. The underlying interface also
accepts consecutive chunks, but buffering a chunk is not required. Preserve
batch stream order when carrying state. Use `state=None` for an independent
sequence.

Inputs, state, and model must share device and floating-point dtype. Use
`model.to(...)` and move caller-owned tensors explicitly. The encoder does not
silently cast, move, detach, retain, or reset supplied state. Callers control
gradient tracking. `detach_state` keeps values but cuts gradient history;
its returned tensors share storage with the originals.

Normal `train()`, `eval()`, and `state_dict()` behavior is inherited from
PyTorch. Recreate the same input and hidden sizes before loading a state dict.
Standard initialization occurs at construction; units are not cloned and
different specialization is not guaranteed.
