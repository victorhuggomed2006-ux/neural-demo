# Reading numbers from 0 to 1000

A CNN that reads a whole handwritten number (up to four digits) from a strip of four slots
(thousands, hundreds, tens, units), 28 x 112 pixels. Each slot predicts a digit 0-9 or "blank".

## Data
Numbers are built on the fly by pasting real MNIST digits into the slots with random shifts, so the
model sees an endless stream of new numbers. Validation uses digits from the MNIST *test* split,
which the model never trained on.

## Model
`NumberCNN`: three conv blocks (16, 32, 64 channels, 3x3, ReLU, max-pool), dense 128 with dropout,
and one 11-way classifier per slot. 373,164 parameters.

## Result (30 epochs, AMD Radeon RX 6600 through DirectML, about 7 minutes)
Evaluated on 5,000 numbers built from unseen digits:

| Metric | Value |
|---|---|
| Whole number correct | 98.2% |
| Hundreds / tens / units | 99.6% / 99.4% / 99.2% |
| Loss | 0.0145 |

Limits: clean MNIST digits placed in fixed slots. Mouse-drawn numbers will do worse.

## Train
```
python experiments/numbers_0_1000/train.py                      # auto: cuda, then dml, then cpu
python experiments/numbers_0_1000/train.py --epochs 30 --device dml --threads 3
python experiments/numbers_0_1000/train.py --smoke               # quick check
```
`--device dml` needs `torch-directml` (Python 3.12, torch 2.4.1) in a separate environment.

Weights: `artifacts/numbers_0_1000/numbers_cnn.pt` (CC BY-SA 4.0, see `LICENSE-WEIGHTS-AND-DOCS`).