# neural-demo

A small neural network that recognises handwritten digits. It is **trained in Python (PyTorch, CPU only)**
and its weights are exported so the **same network runs in the browser, with no server and no
data leaving your device**. Everything is here: the training code, the weights, the tests and the
documentation of how it was built.

- **99.09%** on the MNIST test set (also 99.09% after rounding the weights to float16)
- **26,698 parameters**, **72 KB** of weights for the browser
- The JavaScript forward pass is written by hand (no libraries) and is **tested against the Python
  reference**: the largest difference is below 0.00001

> The 99.09% is on the standard test set. Digits drawn with a mouse usually score lower.
> Read the [model card](MODEL_CARD.md) for the limitations.

## Try it

Open `web/index.html` in a browser (a double-click is enough), draw a digit, and watch the
probabilities and what each filter detected. Nothing is uploaded.

## Reproduce it

```
uv sync                                      # PyTorch for CPU only
uv run python -m neural_demo.train           # about 4 minutes on a laptop CPU
uv run python -m neural_demo.export          # writes artifacts/weights.js and artifacts/reference.json
node web_test/forward.test.mjs               # JavaScript == Python reference
uv run pytest && uv run ruff check . && uv run mypy && uv run bandit -r src
```

## What is in the repository

| Path | What it is |
|---|---|
| `src/neural_demo/` | data, model, training and export (Python, fully typed) |
| `tests/` | tests for the model, the NumPy reference and the export |
| `artifacts/` | trained weights (`mlp.pt`), browser weights (`weights.js`), reference outputs |
| `web/` | the browser demo and the hand-written JavaScript network |
| `web_test/` | proves the JavaScript reproduces the Python results |
| `MODEL_CARD.md` | what the model is, how it was trained, results and limitations |
| `docs/ENGINEERING.md` | the decisions behind the project |

## Design notes

- Inputs are pixels in [0, 1] with no mean/std normalisation, so the browser does exactly what training did.
- Training uses small random shifts, scale and rotation, because drawn digits are less centred than MNIST.
- Weights are exported as float16 (base64). The reference outputs and the accuracy are computed
  **after** the rounding, so the browser must reproduce them.

## License

- **Code:** GNU Affero General Public License v3.0 or later (`LICENSE`). If you modify and distribute
  it, or offer it as a service over a network, you must publish your changes under the same license.
- **Weights, model card and documentation:** Creative Commons Attribution-ShareAlike 4.0
  (`LICENSE-WEIGHTS-AND-DOCS`). Give credit and keep the same license on derived work.

These licenses allow commercial use (no free-software license can forbid it); what they guarantee is that
whoever receives the work also receives the source and may share it freely.

The model was trained on [MNIST](http://yann.lecun.com/exdb/mnist/); check the dataset terms when reusing it.

Copyright (c) 2026 Victor Huggo de Medeiros.
