"""Generate the documentation images: the landing-page hero and the gallery thumbnails.

Maintainer tool. Run after the notebooks have been executed (``jupytext --execute``):

    python scripts/make_doc_images.py            # hero + all thumbnails
    python scripts/make_doc_images.py --thumbs   # thumbnails only (no pci run)

Thumbnails are taken from the first figure of each executed notebook under
``examples/`` and ``case_studies/`` and written to ``<folder>/thumbs/<notebook>.png``,
so the gallery always shows what the notebook actually produces. The hero is a small
dedicated run of the 4-DOF testbed (unknown stiffness and interface force).
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FOLDERS = ["examples", "case_studies"]
THUMB_WIDTH = 720


def first_figure(notebook: Path) -> bytes | None:
    """Return the PNG bytes of the first image output in ``notebook``."""
    nb = json.loads(notebook.read_text())
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if png:
                return base64.b64decode(png)
    return None


def write_thumbnail(png: bytes, target: Path) -> None:
    """Downscale a figure to the gallery width; fall back to the raw PNG without Pillow."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        target.write_bytes(png)
        return
    im = Image.open(io.BytesIO(png)).convert("RGB")
    if im.width > THUMB_WIDTH:
        im = im.resize((THUMB_WIDTH, round(im.height * THUMB_WIDTH / im.width)), Image.LANCZOS)
    im.save(target, optimize=True)


def make_thumbnails() -> None:
    for folder in FOLDERS:
        for nb in sorted((ROOT / folder).glob("*.ipynb")):
            png = first_figure(nb)
            if png is None:
                print(f"  {nb.relative_to(ROOT)}: no figure output, skipped")
                continue
            target = ROOT / folder / "thumbs" / (nb.stem + ".png")
            write_thumbnail(png, target)
            print(f"  wrote {target.relative_to(ROOT)}  ({target.stat().st_size / 1024:.0f} KB)")


def make_hero() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pci

    chain = pci.MassSpringChain.uniform(4, mass=500.0, k=50_000.0, c=300.0)
    dt, T = 1e-3, 5.0
    loads = {d: {"type": "random", "std": 400.0, "seed": d} for d in range(1, 5)}
    truth = chain.simulate(loads, dt=dt, T=T, x0={"x1": 0.01, "v1": 0.01})
    data = chain.measure(truth, ["a1", "a4"], noise_std=1e-3, seed=123)
    partition = [[1, 2], [3, 4]]
    system = chain.decompose(partition, unknowns={"k4": {"initial": 30_000.0, "std": 50_000.0}},
                             sensors=["a1", "a4"], noise_std=1e-3, filters="ukf", schedule="jacobi",
                             x0={"x1": 0.01, "v1": 0.01}, state_var=1e-4, process_var=1e-18, r_inflation=100.0)
    truth_dict = {**truth.as_dict(), **chain.interface_forces(truth, partition)}
    res = system.estimate(data, loads={f"f{d}": v for d, v in loads.items()}, dt=dt, T=T, truth=truth_dict)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
    t = res.t
    k4, s = res.state("k4"), res.std("k4")
    ax1.axhline(50_000, color="k", lw=1.2, label="true k4")
    ax1.plot(t, k4, color="#303f9f", lw=1.4, label="S2 estimate")
    ax1.fill_between(t, k4 - 2 * s, k4 + 2 * s, color="#303f9f", alpha=0.18, label="±2σ")
    ax1.set_title("unknown stiffness k4, identified by subsystem S2")
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("k4 [N/m]"); ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.3)
    name = list(res.messages)[0]
    tm = t[:-1]
    sl = slice(int(1.0 / dt), int(1.5 / dt))
    ax2.plot(tm[sl], truth_dict[name][: tm.size][sl], "k", lw=1.2, label="true interface force")
    ax2.plot(tm[sl], res.messages[name][sl], color="#ef6c00", lw=1.2, ls="--", label="message S1 → S2")
    ax2.set_title("the message exchanged on the interface (1.0 to 1.5 s)")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("force [N]"); ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "docs" / "images" / "hero.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.05)
    print(f"  wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    if "--thumbs" not in sys.argv:
        print("hero image")
        make_hero()
    print("thumbnails")
    make_thumbnails()
