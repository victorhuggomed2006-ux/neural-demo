# How it was built (engineering notes)

These notes explain the decisions behind the project, so you can judge them or reuse them.

## 1. Why a CNN

A fully connected network (784 -> 64 -> 32 -> 10) reached about 96% on the test set. Convolutions share
the same small filter across the whole image, so the network needs far fewer parameters to find
strokes and corners wherever they are. The CNN has 26,698 parameters and reaches 99.09%.

## 2. Same result in two languages

The browser must give the same answer as Python. Three rules keep that true:

1. **Same input range.** Pixels are in [0, 1] with no mean/std normalisation, so the browser does exactly
   what training did.
2. **Weights rounded before measuring.** Weights are shipped as float16 (half the size). The reference
   outputs and the reported accuracy are computed **after** rounding, so the browser must reproduce them.
3. **A test that proves it.** `export.py` writes `reference.json` (pixels and logits of 12 MNIST images).
   `web_test/forward.test.mjs` runs the hand-written JavaScript on them and fails if any logit differs by
   more than 0.001 or any digit changes. Today the largest difference is below 0.00001.

The Python side also tests that a plain NumPy forward pass equals PyTorch, so the reference is trustworthy.

## 3. Preparing a drawing like MNIST

MNIST digits are cropped, fitted into a 20 x 20 box, and centred in 28 x 28 by their centre of mass.
The demo does the same to the canvas: it finds the bounding box of the stroke, scales it to 20 pixels
on the longest side, and shifts it so the centre of mass sits at the middle. Without this step, a digit
drawn small or in a corner is misread.

## 4. Quality gates

`ruff` (lint and format), `mypy --strict`, `bandit`, `pytest` with coverage, and `pip-audit`, `gitleaks`
and CodeQL in CI. Training is seeded; rerunning it gives the same weights on the same machine.

## 5. What I would do next

Test-time augmentation, a "not a digit" rejection class, a larger network with the same size budget
(quantisation to 8 bits), and a robustness test with mouse-drawn digits.
