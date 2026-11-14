"""Train a CNN that reads a whole handwritten number from 0 to 1000.

Runs on CPU, NVIDIA (cuda) or AMD/Intel GPUs on Windows (dml), with live terminal output.

The input is a strip of 4 digit slots (thousands, hundreds, tens, units), 28 x 112 pixels.
Numbers are built on the fly by pasting real MNIST digits into the slots with random shifts, so the
model sees an endless stream of new numbers. Each slot predicts 0-9 or "blank" (class 10).

Run (from the neural-demo folder):
    .venv\\Scripts\\python.exe experiments\\numbers_0_1000\\train.py
    .venv\\Scripts\\python.exe experiments\\numbers_0_1000\\train.py --epochs 20 --threads 4
    .venv\\Scripts\\python.exe experiments\\numbers_0_1000\\train.py --smoke      (quick test)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets

ROOT = Path(os.environ.get("NUM_ROOT", str(Path(__file__).resolve().parents[2])))
OUT_DIR = ROOT / "artifacts" / "numbers_0_1000"
SLOTS = 4  # 1000 has four digits
BLANK = 10
CLASSES = 11  # digits 0-9 plus blank
W = 28 * SLOTS


# ----------------------------------------------------------------------------- data
def load_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.MNIST(str(ROOT / "data"), train=train, download=True)
    return ds.data.float().div(255.0), ds.targets


class NumberFactory:
    """Builds random numbers 0..1000 as images, using the digits of one MNIST split."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, seed: int) -> None:
        self.images = images
        self.by_class = [torch.nonzero(labels == c).flatten() for c in range(10)]
        self.gen = torch.Generator().manual_seed(seed)

    def digit(self, d: int) -> torch.Tensor:
        pool = self.by_class[d]
        i = int(pool[int(torch.randint(len(pool), (1,), generator=self.gen))])
        return self.images[i]

    def make(self, batch: int, jitter: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return images (N,1,28,112), slot labels (N,4) and the numbers (N,)."""
        nums = torch.randint(0, 1001, (batch,), generator=self.gen)
        x = torch.zeros(batch, 1, 28, W)
        y = torch.full((batch, SLOTS), BLANK, dtype=torch.long)
        for b in range(batch):
            text = str(int(nums[b]))
            first = SLOTS - len(text)  # numbers are right-aligned, blanks on the left
            for k, ch in enumerate(text):
                slot = first + k
                y[b, slot] = int(ch)
                dx = int(torch.randint(-jitter, jitter + 1, (1,), generator=self.gen))
                dy = int(torch.randint(-2, 3, (1,), generator=self.gen))
                img = torch.roll(self.digit(int(ch)), shifts=(dy, 0), dims=(0, 1))
                lo = slot * 28 + dx
                a, c = max(lo, 0), min(lo + 28, W)
                x[b, 0, :, a:c] = torch.maximum(x[b, 0, :, a:c], img[:, a - lo : c - lo])
        return x, y, nums


# ----------------------------------------------------------------------------- model
class NumberCNN(nn.Module):
    """Conv backbone over the whole strip, then one small classifier per slot."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 x 14 x 56
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 7 x 28
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 3 x 14
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 3 * 14, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.heads = nn.Linear(128, SLOTS * CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape (N, 4, 11)."""
        out: torch.Tensor = self.heads(self.fc(self.features(x)))
        return out.view(-1, SLOTS, CLASSES)


def ascii_strip(img: torch.Tensor) -> str:
    """Draw a (28,112) image in the terminal, 14 rows x 56 columns."""
    small = nn.functional.avg_pool2d(img.view(1, 1, 28, W), 2).view(14, W // 2)
    chars = " .:-=+*#%@"
    return "\n".join(
        "   |" + "".join(chars[min(int(v * 9.99), 9)] for v in row) + "|" for row in small
    )


# ----------------------------------------------------------------------------- training
def evaluate(
    model: NumberCNN, factory: NumberFactory, n: int, device: torch.device
) -> tuple[float, float, list[float]]:
    model.eval()
    correct_all = 0
    slot_ok = torch.zeros(SLOTS, device=device)
    loss_sum = 0.0
    with torch.no_grad():
        for _ in range(n // 250):
            x, y, _nums = factory.make(250, jitter=3)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(
                nn.functional.cross_entropy(logits.reshape(-1, CLASSES), y.reshape(-1))
            )
            pred = logits.argmax(-1)
            slot_ok += (pred == y).float().sum(0)
            correct_all += int((pred == y).all(1).sum())
    total = (n // 250) * 250
    return correct_all / total, loss_sum / (n // 250), [float(v) / total for v in slot_ok]


def pick_device(name: str) -> torch.device:
    """cuda (NVIDIA) > dml (AMD/Intel on Windows via torch-directml) > cpu."""
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("auto", "dml"):
        try:
            import torch_directml  # type: ignore[import-not-found,unused-ignore]

            dev: torch.device = torch_directml.device()
            return dev
        except ImportError:
            if name == "dml":
                raise SystemExit("torch-directml is not installed in this environment") from None
    return torch.device("cpu")


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device.type == "cpu":
        return "cpu"
    import torch_directml  # type: ignore[import-not-found,unused-ignore]

    return f"dml ({torch_directml.device_name(0)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=250, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "dml"],
        help="auto picks the fastest",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.steps = 1, 12
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print(" Reading numbers from 0 to 1000 - CNN training")
    print("=" * 72)
    tr_img, tr_lab = load_mnist(True)
    te_img, te_lab = load_mnist(False)
    train = NumberFactory(tr_img, tr_lab, seed=args.seed)
    valid = NumberFactory(te_img, te_lab, seed=123)  # digits the model never trained on
    device = pick_device(args.device)
    model = NumberCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f" device: {describe(device)} | parameters: {n_params:,} | slots: {SLOTS}"
        f" | threads: {args.threads} | batch: {args.batch}"
    )
    print(f" steps/epoch: {args.steps} | epochs: {args.epochs} | lr: {args.lr}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)
    best = 0.0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_ok = 0.0, 0.0
        te = time.time()
        for step in range(1, args.steps + 1):
            x, y, _nums = train.make(args.batch)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, CLASSES), y.reshape(-1))
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            sched.step()
            run_loss += loss.item()
            run_ok += float((logits.argmax(-1) == y).all(1).float().mean())
            if step % 10 == 0 or step == args.steps:
                bar = "#" * int(30 * step / args.steps)
                print(
                    f"\r epoch {epoch:>2}/{args.epochs} [{bar:<30}] step {step:>3}/{args.steps}"
                    f"  loss {run_loss / step:.4f}  number acc {run_ok / step:6.1%}",
                    end="",
                    flush=True,
                )
        acc, vloss, per_slot = evaluate(model, valid, 2000 if not args.smoke else 250, device)
        mark = ""
        if acc > best:
            best = acc
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items()}, OUT_DIR / "numbers_cnn.pt"
            )
            mark = "  <- saved (best)"
        done = (time.time() - t0) / epoch * (args.epochs - epoch)
        print(
            f"\n   validation (unseen digits): whole number {acc:6.1%} | loss {vloss:.4f}"
            f" | thousands {per_slot[0]:.1%} hundreds {per_slot[1]:.1%}"
            f" tens {per_slot[2]:.1%} units {per_slot[3]:.1%}"
            f"\n   epoch time {time.time() - te:5.1f}s | time left ~{done / 60:4.1f} min{mark}"
        )
        model.eval()
        with torch.no_grad():
            x, y, nums = valid.make(1)
            pred = model(x.to(device)).argmax(-1).cpu()
        guess = "".join("" if d == BLANK else str(d) for d in pred[0].tolist()) or "?"
        print(ascii_strip(x[0, 0]))
        verdict = "OK" if guess == str(int(nums[0])) else "wrong"
        print(f"   real number: {int(nums[0])}   model read: {guess}   {verdict}\n")

    print("=" * 72)
    print(f" done in {(time.time() - t0) / 60:.1f} min | best whole-number accuracy: {best:.1%}")
    print(f" model saved at: {OUT_DIR / 'numbers_cnn.pt'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
