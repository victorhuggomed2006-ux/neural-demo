# Model card: digit-cnn

A small convolutional network that recognises handwritten digits (0 to 9). It is a teaching and
portfolio model, built to show the whole path: train, verify, export and run **on the device**.

## Model

| | |
|---|---|
| Type | Convolutional neural network (CNN), image classification |
| Input | One 28 x 28 grey-scale image, pixel values in [0, 1] (white digit on black) |
| Output | 10 logits, one per digit; a softmax turns them into probabilities |
| Parameters | 26,698 |
| Layers | conv 3x3 (8) + ReLU + max-pool, conv 3x3 (16) + ReLU + max-pool, dense 32 + ReLU, dense 10 |
| Weights | `artifacts/mlp.pt` (PyTorch, kept for reproducibility) and `artifacts/weights.js` (float16, 72 KB) |
| Framework | PyTorch (CPU), exported to a hand-written JavaScript forward pass |

## Intended use

- Learning and demonstration: see how a small network works, inspect what each filter detects.
- Running a model in the browser with no server and no data leaving the device.

**Not intended for:** reading documents, cheques, medical or legal forms, or any decision that
affects people. It has not been tested for those uses.

## Training data

[MNIST](http://yann.lecun.com/exdb/mnist/) (Y. LeCun, C. Cortes, C. Burges): 60,000 training and
10,000 test images of handwritten digits, derived from NIST data. Check the dataset terms before
reusing it. Training images get small random shifts (8%), scale (90 to 110%) and rotation (8 degrees)
to make the model less sensitive to where and how large the digit is.

## Results

| Measure | Value |
|---|---|
| MNIST test accuracy, float32 | 99.09% |
| MNIST test accuracy, float16 weights (what the browser uses) | 99.09% |
| Difference between the JavaScript and the Python logits on 12 reference images | below 0.00001 |

Training: 14 epochs, Adam (lr 3e-3, cosine decay), batch 128, seed 0, CPU only (about 4 minutes).

## Limitations (please read)

- The 99.09% is on the **standard MNIST test set**. Digits drawn with a mouse or a finger look
  different and the accuracy is usually **lower**. The demo centres and scales the drawing like MNIST
  does, which helps but does not remove the gap.
- The network only knows 10 classes. Given a letter, a symbol or a scribble, it still answers with a
  digit, and often with high confidence. It has no "I do not know" option.
- It can be fooled by small, deliberate changes to an image (adversarial examples).
- Digits written in styles that are rare in MNIST (for example a crossed 7 or a 1 with a base) may be
  misread.

## How to reproduce

```
uv sync
uv run python -m neural_demo.train      # about 4 minutes on a laptop CPU
uv run python -m neural_demo.export     # writes artifacts/weights.js and artifacts/reference.json
node web_test/forward.test.mjs          # the JavaScript matches the Python reference
```

## License

Code: GNU AGPL-3.0-or-later (`LICENSE`). Weights, this card and the documentation:
CC BY-SA 4.0 (`LICENSE-WEIGHTS-AND-DOCS`). Copyright (c) 2026 Victor Huggo de Medeiros.
