import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import pandas as pd

# ---------- utilities ----------
def circular_median(angles):
    """Circular median: minimizer of sum of absolute angular deviations."""
    a = np.asarray(angles, float)

    grid = np.linspace(-np.pi, np.pi, 721, endpoint=False)  # 0.5° grid
    def cost(phi): return np.sum(np.abs(np.angle(np.exp(1j*(a - phi)))))
    c = np.array([cost(g) for g in grid])
    phi0 = grid[np.argmin(c)]

    lo, hi = phi0 - np.deg2rad(1), phi0 + np.deg2rad(1)
    for _ in range(40):
        g1 = lo + 0.382*(hi - lo); g2 = lo + 0.618*(hi - lo)
        if cost(g1) < cost(g2): hi = g2
        else: lo = g1
    return (lo + hi) / 2.0

def pca_theta_and_width_ellipse(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return 0.0, 0.0

    X = np.column_stack([x, y])
    Xc = X - X.mean(axis=0)
    C = np.cov(Xc, rowvar=False)
    evals, evecs = np.linalg.eigh(C)            # ascending
    lam1, lam2 = evals[-1], evals[-2]
    v1 = evecs[:, -1]
    theta = float(np.arctan2(v1[1], v1[0]))

    # flip θ to point into data (use mean vector)
    mu = X.mean(axis=0)
    if mu[0]*np.cos(theta) + mu[1]*np.sin(theta) < 0:
        theta += np.pi

    # ellipse width (clamped)
    half = np.arctan(np.sqrt(max(lam2, 0.0) / max(lam1, 1e-12)))
    width = float(np.clip(2.0*half, 0.0, np.pi - 1e-6))
    return theta, width

def per_mouse_theta_width(df, xcol, ycol, mouse_col):
    rows = []
    for m, g in df.groupby(mouse_col, sort=False):
        theta, width = pca_theta_and_width_ellipse(g[xcol], g[ycol])
        rows.append({"mouse": m, "theta": theta, "width": width})
    per_mouse = pd.DataFrame(rows)
    # aggregate across mice
    theta_group = circular_median(per_mouse["theta"].to_numpy())
    width_group = float(np.mean(per_mouse["width"].to_numpy()))
    return per_mouse, theta_group, width_group

# ---------- drawing (anchored at a tip; default tip=(0,0)) ----------
def draw_cone(ax, tip, theta, width, L=None, color="r", lw=2, edge_ls="-", edge_lw=1, fill_alpha=0.10, x=None, y=None):
    """
    Draw a cone from 'tip' with central angle 'theta' and full 'width' (all radians).
    If L is None, estimate from data radii to 95th percentile using x,y relative to tip.
    """
    x0, y0 = map(float, tip)
    width = float(np.clip(width, 0.0, np.pi - 1e-6))
    half = width / 2.0

    # auto length if data given
    if L is None and x is not None and y is not None:
        xr = np.asarray(x, float) - x0
        yr = np.asarray(y, float) - y0
        r = np.sqrt(xr**2 + yr**2)
        L = float(np.nanpercentile(r, 95)) if np.any(np.isfinite(r)) else 1.0
    elif L is None:
        L = 1.0

    # central ray
    x1, y1 = x0 + L*np.cos(theta), y0 + L*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls="--")

    # edge rays
    for ang in (theta - half, theta + half):
        xe, ye = x0 + L*np.cos(ang), y0 + L*np.sin(ang)
        ax.plot([x0, xe], [y0, ye], color=color, ls=edge_ls, lw=edge_lw)

    # wrap-safe wedge
    start_deg = (np.degrees(theta - half)) % 360.0
    span_deg  = np.degrees(width)
    wedge = Wedge((x0, y0), L, start_deg, start_deg + span_deg,
                  facecolor=color, alpha=fill_alpha, edgecolor=None)
    ax.add_patch(wedge)
#%%
sids = sessions_ctx#SESSIONS_ALL[1:4]
dfb = ctx_mice.copy()

s1 =f"int_optimized_{sids[0]}_rstd_common" ""#"int_optimized_ctx_rstd"   
s2 =f"int_optimized_{sids[1]}_rstd_common" #"int_optimized_landmark1_rstd"  
s3 = f"int_optimized_{sids[2]}_rstd_common"#"int_optimized_landmark2_rstd"  

fig, ax = plt.subplots(figsize=(5, 5))
mask1 = dfb["detected_in_sessions"].apply(
    lambda s: (utils.in_s(s, sids[0]) and
               utils.in_s(s, sids[1]) and
               (not utils.in_s(s, sids[2])))
)
mask2 = dfb["detected_in_sessions"].apply(
    lambda s: (utils.in_s(s, sids[0]) and
               utils.in_s(s, sids[1]))
)
spec2 = dfb.loc[mask2].copy()

stats_all = (spec2.groupby(["mouse"])
           .apply(lambda g: pd.Series(cone_polar_stats(g[s1], g[s2]),
                                      index=["theta","width"]))
           .reset_index())



spec = dfb.loc[mask1].copy()

#%%

# ---------- example usage ----------
# df must have columns: mouse_col, xcol (=S1), ycol (=S2)
per_mouse, theta_med, width_med = per_mouse_theta_width(spec2, xcol=s1, ycol=s2, mouse_col="mouse")
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(spec2[s1], spec2[s2], s=2, c="tab:blue")
draw_cone(ax, tip=(-2.0, -2.0), theta=theta_med, width=width_med, L=20,
          color="navy", x=spec2[s1], y=spec2[s2])
ax.set_aspect("equal", adjustable="box")

per_mouse, theta_med, width_med = per_mouse_theta_width(spec, xcol=s1, ycol=s2, mouse_col="mouse")
#fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(spec[s1], spec[s2], s=2, alpha=0.8, c="y")
draw_cone(ax, tip=(-2.0, -2.0), theta=theta_med, width=width_med, L=20,
          color="green", x=spec[s1], y=spec[s2])
ax.set_aspect("equal", adjustable="box")

ax.legend(scatterpoints=1, markerscale=7.0, fontsize=10)
plt.xlim(-5,30)
plt.ylim(-5,30)
plt.xlabel("Intensywność S1", fontsize=14)
plt.ylabel("Intensywność S2", fontsize=14)
plt.title("Komórki selektywne na tle populaci - S2S3", fontsize=18, pad=14, color="white")

plt.show()

#%%
fig, ax = plt.subplots(figsize=(5, 5))
mask1 = dfb["detected_in_sessions"].apply(
    lambda s: (utils.in_s(s, sids[1]) and
               utils.in_s(s, sids[2]) and
               (not utils.in_s(s, sids[0])))
)
mask2 = dfb["detected_in_sessions"].apply(
    lambda s: (utils.in_s(s, sids[2]) and
               utils.in_s(s, sids[1]))
)
spec2 = dfb.loc[mask2].copy()

stats_all = (spec2.groupby(["mouse"])
           .apply(lambda g: pd.Series(cone_polar_stats(g[s2], g[s3]),
                                      index=["theta","width"]))
           .reset_index())



spec = dfb.loc[mask1].copy()

# ---------- example usage ----------
# df must have columns: mouse_col, xcol (=S1), ycol (=S2)
per_mouse, theta_med, width_med = per_mouse_theta_width(spec2, xcol=s2, ycol=s3, mouse_col="mouse")

ax.scatter(spec2[s2], spec2[s3], s=2, c="tab:blue")
draw_cone(ax, tip=(-2.0, -2.0), theta=theta_med, width=width_med, L=20,
          color="navy", x=spec2[s2], y=spec2[s3])
ax.set_aspect("equal", adjustable="box")

per_mouse, theta_med, width_med = per_mouse_theta_width(spec, xcol=s2, ycol=s3, mouse_col="mouse")
#fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(spec[s2], spec[s3], s=2, alpha=0.8, c="y", label="KOMÓRKI\nSELEKTYWNE")
draw_cone(ax, tip=(-2.0, -2.0), theta=theta_med, width=width_med, L=20,
          color="green", x=spec[s2], y=spec[s3])
ax.set_aspect("equal", adjustable="box")
ax.legend(scatterpoints=1, markerscale=7.0, fontsize=10)
plt.xlim(-5,30)
plt.ylim(-5,30)
plt.xlabel("Intensywność S2", fontsize=14)
plt.ylabel("Intensywność S3", fontsize=14)
plt.title("Komórki selektywne na tle populaci - S2S3", fontsize=18, pad=14, color="white")
plt.show()

#%%


fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(spec2[s1],spec2[s2], s=1, color="tab:blue")

# theta, width = cone_polar_stats(spec2[s1], spec2[s2], anchor=(0.0, 0.0), q=0.95)
# draw_cone(ax, spec2[s1], spec2[s2], theta, width, width_unit="deg",
#           anchor=(0.0, 0.0), color="red", L=30, fill=True)

tip, theta, width, L = cone_from_cloud(spec2[s1], spec2[s2], q_tip=0.05, q_width=0.95)
draw_cone(ax, tip, theta, width, L, color="red", lw=2, edge_ls="--", edge_lw=1, fill_alpha=0.10)

ax.set_xlabel("S1 intensity")
ax.set_ylabel("S2 intensity")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Cone overlay")
plt.show()