#!/usr/bin/env python3
"""
Regenerate the data plots of the PC-CSG paper as PNG files (300 DPI).

The LaTeX manuscript draws the same plots natively with PGFPlots, so this
script is only needed if you want raster copies (slides, posters, README).
Every number below is copied from our measurements (Tables 4-8 of the paper
and checkpoints/training_history.json) or computed analytically from the
PIKV constants in pc_csg/config.py. Nothing is simulated.

Usage:
    python generate_plots.py            # writes PNGs into ./images/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(exist_ok=True)

MU, G, L = 0.7, 9.81, 2.7
MU_G = MU * G  # 6.867 m/s^2
TAU_CRIT, LAMBDA = 3.0, 0.5


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT / name}")


# Fig. 1: braking distance vs distance covered in one second
def fig_intro():
    v_kmh = np.linspace(0, 150, 151)
    v = v_kmh / 3.6
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(v_kmh, v**2 / (2 * MU_G), label=r"Braking distance $v^2/(2\mu g)$, $\mu=0.7$")
    ax.plot(v_kmh, v, "--", color="tab:red", label="Distance covered in 1 s")
    ax.plot([100, 100], [(100 / 3.6) ** 2 / (2 * MU_G), 100 / 3.6], "ko")
    ax.set_xlabel("Vehicle speed (km/h)")
    ax.set_ylabel("Distance (m)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 140)
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "intro_braking.png")


# Fig. 3: training history (checkpoints/training_history.json)
def fig_training():
    epochs = [1, 2, 3, 4, 5]
    train = [0.3797, 0.0738, 0.0506, 0.0450, 0.0450]
    val = [0.0840, 0.0454, 0.0409, 0.0398, 0.0396]
    mae = [0.0440, 0.0417, 0.0423, 0.0422, 0.0417]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.plot(epochs, train, "o-", label="Train loss")
    a.plot(epochs, val, "s-", color="tab:red", label="Val. loss")
    a.set(xlabel="Epoch", ylabel="Loss", title="(a) Training and validation loss")
    a.legend()
    a.grid(alpha=0.3)
    b.plot(epochs, mae, "^-", color="teal")
    b.set(xlabel="Epoch", ylabel="Val. CRT MAE", title="(b) Validation CRT MAE")
    b.grid(alpha=0.3)
    save(fig, "training_curves.png")


# Fig. 4: component latency on RTX 3050 Laptop GPU (Table 4)
def fig_latency():
    names = ["Scene Graph", "CSGA Attention", "Bicycle Rollout", "PIKV Validator", "CRT Arbiter"]
    ms = [0.33, 1.57, 4.12, 0.53, 0.13]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.barh(names[::-1], ms[::-1], color="tab:blue")
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Total {sum(ms):.2f} ms ({1000 / sum(ms):.1f} FPS)")
    ax.set_xlim(0, 5)
    ax.grid(axis="x", alpha=0.3)
    save(fig, "latency_breakdown.png")


# Fig. 5: PIKV pruning per mode (Table 5) + analytic kinematic envelope
def fig_pikv():
    modes = ["Brake", "Swerve L", "Swerve R", "Accel.", "Lane dep.", "Stationary"]
    rate = [0.0, 70.8, 21.2, 100.0, 73.5, 10.6]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    bars = a.bar(modes, rate, color="tab:orange")
    a.bar_label(bars, fmt="%.1f", fontsize=7)
    a.axhline(46.0, ls="--", color="red", label="overall 46.0% (312/678)")
    a.set(ylabel="Hypotheses rejected (%)", ylim=(0, 115), title="(a) Pruning rate per mode")
    a.tick_params(axis="x", rotation=30)
    a.legend(fontsize=8)

    v = np.linspace(0, 20, 201)
    for delta, style, label in [(0.45, "-", r"Swerve prior $\delta=0.45$"),
                                (0.15, "-.", r"Lane-dep. prior $\delta=0.15$")]:
        b.plot(v, v**2 * np.tan(delta) / L, style, label=label)
        v_cross = np.sqrt(MU_G * L / np.tan(delta))
        b.plot(v_cross, MU_G, "ko", ms=4)
    b.axhline(MU_G, ls="--", color="red", label=r"$\mu g = 6.867$")
    b.axhline(6.0, ls=":", color="gray", label=r"$a_{lat,max} = 6.0$")
    b.set(xlabel="Speed v (m/s)", ylabel=r"$a_{lat} = v^2 \tan\delta / L$ (m/s$^2$)",
          xlim=(0, 20), ylim=(0, 20), title="(b) Kinematic envelope")
    b.legend(fontsize=7, loc="upper left")
    b.grid(alpha=0.3)
    save(fig, "pikv_analysis.png")


# Fig. 6: demo scenarios A-D (Table 8) + hazard sigmoid
def fig_scenarios():
    crt = [0.000, 0.103, 0.894, 0.987]
    ttc = [33.3, 4.1, 1.9, 0.5]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    bands = [(0, 0.25, "#d9f2d9", "L0"), (0.25, 0.50, "#d6f0f5", "L1"),
             (0.50, 0.70, "#fff5c2", "L2"), (0.70, 0.85, "#ffe0c2", "L3"),
             (0.85, 1.08, "#f8d0d0", "L4")]
    for lo, hi, col, name in bands:
        a.axhspan(lo, hi, color=col, zorder=0)
        a.text(-0.45, (lo + hi) / 2, name, fontsize=7, va="center")
    bars = a.bar(["A", "B", "C", "D"], crt, color="tab:blue", zorder=2)
    a.bar_label(bars, fmt="%.3f", fontsize=7)
    a.set(ylabel="CRT", ylim=(0, 1.08), title="(a) Measured CRT per scenario")

    t = np.linspace(0.01, 6, 300)
    b.plot(t, 1 / (1 + np.exp((t - TAU_CRIT) / LAMBDA)), "k-", label=r"$H(TTC)$")
    b.plot(ttc[1:], crt[1:], "ro", label="Measured CRT (B, C, D)")
    for name, x, y in zip(["B", "C", "D"], ttc[1:], crt[1:]):
        b.annotate(name, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    b.axvline(TAU_CRIT + LAMBDA * np.log(3), ls="--", color="gray")
    b.set(xlabel="Worst-case TTC (s)", ylabel="H(TTC) / CRT", xlim=(0, 6), ylim=(0, 1.05),
          title="(b) CRT vs. hazard curve")
    b.legend(fontsize=8)
    b.grid(alpha=0.3)
    save(fig, "scenarios_crt.png")


if __name__ == "__main__":
    fig_intro()
    fig_training()
    fig_latency()
    fig_pikv()
    fig_scenarios()
