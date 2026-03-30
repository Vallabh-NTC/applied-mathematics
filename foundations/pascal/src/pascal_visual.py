import math
import numpy as np
import matplotlib

matplotlib.use("TkAgg")   # interactive window

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =========================
# Settings
# =========================
NUM_ROWS = 100
USE_LOG_SCALE = True   # keep False if you want true z = C(n,r)
COLOR_CYCLE = 10        # repeat colors every 10 rows


# =========================
# Build Pascal 3D points
# =========================
x_vals = []
y_vals = []
z_vals = []
row_colors = []

for n in range(NUM_ROWS):
    color_id = n % COLOR_CYCLE

    for r in range(n + 1):
        coeff = math.comb(n, r)

        x_exp = n - r
        y_exp = r

        x_vals.append(float(x_exp))
        y_vals.append(float(y_exp))

        if USE_LOG_SCALE:
            z_vals.append(math.log10(coeff))
        else:
            z_vals.append(float(coeff))

        row_colors.append(color_id)


# Convert to numpy arrays
x_vals = np.array(x_vals, dtype=float)
y_vals = np.array(y_vals, dtype=float)
z_vals = np.array(z_vals, dtype=float)
row_colors = np.array(row_colors, dtype=int)


# =========================
# Plot
# =========================
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    x_vals,
    y_vals,
    z_vals,
    c=row_colors,
    cmap="tab10",     # 10 repeating row colors
    s=8,
    depthshade=True
)

ax.set_title(f"Pascal Triangle in 3D ({NUM_ROWS} rows)")
ax.set_xlabel("x exponent = n - r")
ax.set_ylabel("y exponent = r")

if USE_LOG_SCALE:
    ax.set_zlabel("log10(binomial coefficient)")
else:
    ax.set_zlabel("binomial coefficient C(n,r)")

ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()