import math
import matplotlib
matplotlib.use("TkAgg")   # use "QtAgg" if you have PyQt installed

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib as mpl


# ============================================================
# Dark theme setup
# ============================================================
plt.style.use("dark_background")
mpl.rcParams.update({
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "axes.titlecolor": "white",
    "legend.framealpha": 0.25,
})


# ============================================================
# Settings
# ============================================================
PASCAL_MAX_N = 3

# Positive side: 2.5, 5.5, 8.5, ..., 38.5
POSITIVE_ALPHAS = [2.5 + 3 * k for k in range(13)]

# Negative side: -1.5, -2.5, -5.5, -8.5, -11.5, ..., -38.5
NEGATIVE_ALPHAS = [-1.5, -2.5] + [-(2.5 + 3 * k) for k in range(0, 13)]

# Remove duplicates while preserving order
seen = set()
NEGATIVE_ALPHAS = [a for a in NEGATIVE_ALPHAS if not (a in seen or seen.add(a))]

# Extra irrational example
EXTRA_ALPHAS = [math.pi]

NEWTON_ALPHAS = POSITIVE_ALPHAS + NEGATIVE_ALPHAS + EXTRA_ALPHAS

SLIDER_MIN = 5
SLIDER_MAX = 60
INITIAL_TERMS = 5


# ============================================================
# Color map helper
# ============================================================
def make_curve_colors(n):
    cmap = plt.cm.turbo
    if n == 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


CURVE_COLORS = make_curve_colors(len(NEWTON_ALPHAS))


# ============================================================
# Newton generalized binomial coefficient
# ============================================================
def newton_binom(alpha, r):
    if r == 0:
        return 1.0

    value = 1.0
    for k in range(r):
        value *= (alpha - k) / (k + 1)
    return value


# ============================================================
# Signed log transform
# Z = sign(c) * log10(1 + |c|)
# ============================================================
def signed_log10(c):
    if c == 0:
        return 0.0
    return math.copysign(math.log10(1.0 + abs(c)), c)


# ============================================================
# Format alpha for legend
# ============================================================
def format_alpha(alpha):
    if abs(alpha - math.pi) < 1e-12:
        return "π"
    return f"{alpha:.3g}"


# ============================================================
# Draw everything
# ============================================================
def draw_plot(ax, newton_terms):
    ax.cla()
    ax.set_facecolor("black")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.tick_params(axis="z", colors="white")

    try:
        ax.xaxis.pane.set_color((0, 0, 0, 1))
        ax.yaxis.pane.set_color((0, 0, 0, 1))
        ax.zaxis.pane.set_color((0, 0, 0, 1))
    except Exception:
        pass

    # ========================================================
    # 1. Pascal cone (fixed background)
    # ========================================================
    pascal_x, pascal_y, pascal_z = [], [], []

    for n in range(PASCAL_MAX_N + 1):
        row_x, row_y, row_z = [], [], []

        for r in range(n + 1):
            coeff = math.comb(n, r)
            z_val = signed_log10(coeff)

            x_exp = n - r
            y_exp = r

            pascal_x.append(x_exp)
            pascal_y.append(y_exp)
            pascal_z.append(z_val)

            row_x.append(x_exp)
            row_y.append(y_exp)
            row_z.append(z_val)

            ax.text(
                x_exp, y_exp, z_val + 0.05,
                f"{coeff}",
                fontsize=8,
                color="white"
            )

        #ax.plot(
        #    row_x, row_y, row_z,
        #    linewidth=1.6,
        #    color="cyan",
        #    alpha=0.85
        #)

    ax.scatter(
        pascal_x, pascal_y, pascal_z,
        s=75,
        color="cyan",
        label="Pascal (n = 0..3)"
    )

    # ========================================================
    # 2. Newton rows with slider-controlled term count
    # ========================================================
    all_x = pascal_x[:]
    all_y = pascal_y[:]
    all_z = pascal_z[:]

    for alpha, color in zip(NEWTON_ALPHAS, CURVE_COLORS):
        newton_x, newton_y, newton_z = [], [], []

        for r in range(newton_terms):
            coeff = newton_binom(alpha, r)
            z_val = signed_log10(coeff)

            x_exp = alpha - r
            y_exp = r

            newton_x.append(x_exp)
            newton_y.append(y_exp)
            newton_z.append(z_val)

            all_x.append(x_exp)
            all_y.append(y_exp)
            all_z.append(z_val)

            if r < 5:
                z_offset = 0.08 if z_val >= 0 else -0.12
                ax.text(
                    x_exp, y_exp, z_val + z_offset,
                    f"{coeff:.3g}",
                    fontsize=7,
                    color=color
                )

        #ax.plot(
        #    newton_x, newton_y, newton_z,
        #    linewidth=1.6,
        #    color=color,
        #    alpha=0.92
        #)

        ax.scatter(
            newton_x, newton_y, newton_z,
            s=34,
            color=color,
            label=fr"Newton $\alpha = {format_alpha(alpha)}$"
        )

    # ========================================================
    # Axes
    # ========================================================
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)

    z_pad = max(0.3, 0.08 * (z_max - z_min if z_max > z_min else 1.0))

    ax.set_title(
        "Pascal Cone (n ≤ 3) vs many Newton Binomial Rows\n"
        f"first {newton_terms} Newton terms"
    )
    ax.set_xlabel("x exponent (n-r or α-r)")
    ax.set_ylabel("y exponent (r)")
    ax.set_zlabel("sign(c) · log10(1 + |c|)")

    ax.set_xlim(x_min - 1.0, x_max + 1.0)
    ax.set_ylim(y_min - 0.5, y_max + 0.8)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)

    ax.view_init(elev=24, azim=38)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=6, ncol=3)


# ============================================================
# Figure + axis + slider
# ============================================================
fig = plt.figure(figsize=(17, 10))
fig.patch.set_facecolor("black")

ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.16)

draw_plot(ax, INITIAL_TERMS)

slider_ax = plt.axes([0.18, 0.05, 0.64, 0.04], facecolor="dimgray")
term_slider = Slider(
    ax=slider_ax,
    label="Number of Newton series terms",
    valmin=SLIDER_MIN,
    valmax=SLIDER_MAX,
    valinit=INITIAL_TERMS,
    valstep=1,
    color="deepskyblue"
)


# ============================================================
# Slider callback
# ============================================================
def update(val):
    newton_terms = int(term_slider.val)
    draw_plot(ax, newton_terms)
    fig.canvas.draw_idle()


term_slider.on_changed(update)

plt.show()