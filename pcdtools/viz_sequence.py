# scripts/pcdtools/viz_sequence.py
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np

# Minimal, headless-friendly stack
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - ensure 3D projection available

from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


# ----------------------------- Color helpers -----------------------------

def _as_rgba(color: Union[str, Tuple[float, float, float], Tuple[float, float, float, float]], alpha: float = 1.0):
    """Convert a Matplotlib color string or tuple into RGBA tuple."""
    import matplotlib.colors as mcolors
    c = mcolors.to_rgba(color)
    return (c[0], c[1], c[2], c[3] * alpha)


def labels_to_colors(labels: np.ndarray, seed: int = 0, noise_label: int = -1) -> np.ndarray:
    """
    Map integer labels -> RGB colors (Nx4 RGBA array). Noise goes to gray.
    """
    labels = labels.astype(int, copy=False)
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)

    # Palette seeds
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    lut: Dict[int, Tuple[float, float, float, float]] = {}
    next_idx = 0
    for lb in uniq:
        if lb == noise_label:
            lut[lb] = _as_rgba("#999999", alpha=0.80)
            continue
        color = base[next_idx] if next_idx < len(base) else tuple(rng.random(3))
        next_idx += 1
        lut[lb] = _as_rgba(color, alpha=0.95)

    out = np.empty((labels.shape[0], 4), dtype=float)
    for i, lb in enumerate(labels):
        out[i] = lut[lb]
    return out


def mask_to_colors(
    n: int,
    keep_mask: np.ndarray,
    keep_color: Union[str, Tuple[float, float, float]] = "#4c78a8",
    drop_color: Union[str, Tuple[float, float, float]] = "#e45756",
    keep_alpha: float = 0.95,
    drop_alpha: float = 0.95,
) -> np.ndarray:
    """
    Convert a boolean keep_mask into RGBA colors for N points.
    """
    assert keep_mask.shape[0] == n
    colors = np.empty((n, 4), dtype=float)
    kc = _as_rgba(keep_color, alpha=keep_alpha)
    dc = _as_rgba(drop_color, alpha=drop_alpha)
    colors[keep_mask] = kc
    colors[~keep_mask] = dc
    return colors


# ----------------------------- Geometry helpers -----------------------------

def _axes_equal_3d(ax):
    """Set 3D axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


def _fit_plane_svd(pts: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit plane to points with SVD.
    Returns (normal n, d, centroid c) such that n.x + d = 0 at the plane.
    """
    c = pts.mean(axis=0)
    Q = pts - c
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1, :]  # smallest singular vector
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ c)
    return n, d, c


def _plane_extent_from_points(pts: np.ndarray, n: np.ndarray, c: np.ndarray, scale: float = 1.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a small rectangular plane patch aligned with the fitted plane, sized by the data spread.
    Returns meshgrid arrays (X,Y,Z) shaped (m,m).
    """
    # Orthonormal basis (u,v) on the plane
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u); v /= (np.linalg.norm(v) + 1e-12)

    # Project points onto (u, v) to estimate extents
    P = pts - c
    su = P @ u
    sv = P @ v
    ru = (su.max() - su.min()) * 0.5 * scale
    rv = (sv.max() - sv.min()) * 0.5 * scale

    m = 12  # grid resolution for surface
    uu = np.linspace(-ru, +ru, m)
    vv = np.linspace(-rv, +rv, m)
    UU, VV = np.meshgrid(uu, vv)
    XYZ = c[None, None, :] + UU[..., None]*u[None, None, :] + VV[..., None]*v[None, None, :]
    return XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]


# ----------------------------- Recorder -----------------------------

@dataclass
class Camera:
    elev: float = 18.0
    azim: float = -60.0
    dist_pad: float = 1.05  # how much to pad the auto limits


class PCSequenceRecorder:
    """
    Record a sequence of 3D scatter frames and export to GIF/PNGs.

    - Headless Matplotlib renderer
    - Consistent camera across frames (unless overridden)
    - Optional text notes overlay per frame
    - Downsampling for very large point clouds
    """
    def __init__(
        self,
        width: int = 1200,
        height: int = 900,
        dpi: int = 110,
        max_points: int = 200_000,
        background: str = "white",
        camera: Optional[Camera] = None,
        seed: int = 0,
    ):
        self.width = width
        self.height = height
        self.dpi = dpi
        self.max_points = max_points
        self.background = background
        self.camera = camera or Camera()
        self.frames: List[Image.Image] = []
        self._rng = np.random.default_rng(seed)
        # Sticky bounds so the camera doesn't jump
        self._sticky_bounds = None  # (xmin, xmax, ymin, ymax, zmin, zmax)

    # -------------------- public API --------------------

    def add_points(
        self,
        title: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        point_size: float = 1.5,
        notes: Optional[Sequence[str]] = None,
        camera: Optional[Camera] = None,
        downsample: bool = True,
        show_axes: bool = False,
    ):
        """
        Add a generic scatter frame.
        colors: None or (N,4) RGBA in [0,1].
        """
        assert points.ndim == 2 and points.shape[1] == 3
        pts, cols = self._prep_data(points, colors, downsample)

        fig = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        fig.patch.set_facecolor(self.background)
        ax.set_facecolor(self.background)

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=cols, marker=".", depthshade=False)

        # Nice bounds & camera
        self._apply_camera(ax, pts, camera)

        if not show_axes:
            ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=6)

        # Render to image
        buf = io.BytesIO()
        plt.tight_layout(pad=0.1)
        fig.canvas.draw()
        fig.savefig(buf, format="png", dpi=self.dpi, facecolor=self.background, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        img = Image.open(buf).convert("RGBA")
        if notes:
            img = self._overlay_notes(img, notes)
        self.frames.append(img)

    def add_mask_view(
        self,
        title: str,
        points: np.ndarray,
        keep_mask: np.ndarray,
        point_size: float = 1.5,
        notes: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        cols = mask_to_colors(points.shape[0], keep_mask)
        self.add_points(title, points, colors=cols, point_size=point_size, notes=notes, **kwargs)

    def add_labels_view(
        self,
        title: str,
        points: np.ndarray,
        labels: np.ndarray,
        noise_label: int = -1,
        point_size: float = 1.5,
        notes: Optional[Sequence[str]] = None,
        seed: int = 0,
        **kwargs,
    ):
        cols = labels_to_colors(labels, seed=seed, noise_label=noise_label)
        self.add_points(title, points, colors=cols, point_size=point_size, notes=notes, **kwargs)

    def add_plane_fit_view(
        self,
        title: str,
        points: np.ndarray,
        inlier_mask: np.ndarray,
        plane_params: Optional[Tuple[np.ndarray, float]] = None,  # (n, d)
        color_in: Union[str, Tuple[float, float, float]] = "#2ca02c",
        color_out: Union[str, Tuple[float, float, float]] = "#d62728",
        residuals_as_color: bool = False,
        point_size: float = 1.5,
        notes: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        pts = points
        assert pts.ndim == 2 and pts.shape[1] == 3
        assert inlier_mask.shape[0] == pts.shape[0]

        P_in = pts[inlier_mask]
        if plane_params is None:
            n, d, c = _fit_plane_svd(P_in)
        else:
            n, d = plane_params
            # Recompute centroid for plane patch sizing
            c = P_in.mean(axis=0)

        # Plane patch
        X, Y, Z = _plane_extent_from_points(P_in, n, c)

        # Residuals if desired
        cols = np.empty((pts.shape[0], 4), dtype=float)
        if residuals_as_color:
            # signed distances
            dist = (pts @ n + d)
            # robust scale for color normalization
            med = np.median(np.abs(dist))
            scale = 3.0 * (med + 1e-12)
            t = np.clip(0.5 + 0.5 * dist / scale, 0.0, 1.0)  # map to [0,1]
            cmap = plt.get_cmap("coolwarm")
            cols[:] = [*cmap(0.5)]
            cols[:, :] = cmap(t)
            cols[~inlier_mask] = _as_rgba(color_out, alpha=0.95)
        else:
            cols[inlier_mask] = _as_rgba(color_in, alpha=0.95)
            cols[~inlier_mask] = _as_rgba(color_out, alpha=0.95)

        # Draw
        fig = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor(self.background)
        ax.set_facecolor(self.background)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.20, linewidth=0.0, antialiased=True, shade=False, color="#555555")

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=cols, marker=".", depthshade=False)

        # Camera
        self._apply_camera(ax, pts, kwargs.get("camera", None))
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=6)

        buf = io.BytesIO()
        plt.tight_layout(pad=0.1)
        fig.canvas.draw()
        fig.savefig(buf, format="png", dpi=self.dpi, facecolor=self.background, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        img = Image.open(buf).convert("RGBA")
        if notes:
            img = self._overlay_notes(img, notes)
        self.frames.append(img)

    def save_gif(self, path: str, fps: int = 2, loop: int = 0):
        assert len(self.frames) > 0, "No frames added."
        duration = 1.0 / max(1, fps)
        imageio.mimsave(path, [np.array(im) for im in self.frames], duration=duration, loop=loop)

    def save_pngs(self, directory: str, prefix: str = "frame"):
        import os
        os.makedirs(directory, exist_ok=True)
        for i, im in enumerate(self.frames):
            im.save(f"{directory}/{prefix}_{i:03d}.png")

    # -------------------- internals --------------------

    def _prep_data(self, points: np.ndarray, colors: Optional[np.ndarray], downsample: bool):
        pts = points
        if downsample and pts.shape[0] > self.max_points:
            idx = self._rng.choice(pts.shape[0], size=self.max_points, replace=False)
            pts = pts[idx]
            cols = colors[idx] if colors is not None else None
        else:
            cols = colors
        if cols is None:
            cols = np.tile(_as_rgba("#4c78a8", alpha=0.95), (pts.shape[0], 1))
        return pts, cols

    def _apply_camera(self, ax, pts: np.ndarray, camera: Optional[Camera]):
        cam = camera or self.camera
        # Compute global bounds across frames (sticky)
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        if self._sticky_bounds is None:
            pad = (maxs - mins) * (cam.dist_pad - 1.0)
            self._sticky_bounds = (mins - pad, maxs + pad)
        (mins_sticky, maxs_sticky) = self._sticky_bounds
        ax.set_xlim(mins_sticky[0], maxs_sticky[0])
        ax.set_ylim(mins_sticky[1], maxs_sticky[1])
        ax.set_zlim(mins_sticky[2], maxs_sticky[2])
        ax.view_init(elev=cam.elev, azim=cam.azim)
        _axes_equal_3d(ax)

    @staticmethod
    def _overlay_notes(img: Image.Image, notes: Sequence[str]) -> Image.Image:
        """Overlay a translucent note panel with bullet points."""
        if not notes:
            return img
        draw = ImageDraw.Draw(img, "RGBA")
        W, H = img.size
        margin = 16
        panel_w = int(min(W * 0.45, 520))
        panel_h = 22 + 18 * len(notes) + 14
        x0, y0 = margin, margin
        x1, y1 = x0 + panel_w, y0 + panel_h
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 110), outline=(255, 255, 255, 180))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        except Exception:
            font = ImageFont.load_default()
            title_font = font
        draw.text((x0 + 10, y0 + 6), "Notes", font=title_font, fill=(255, 255, 255, 230))
        y = y0 + 28
        for line in notes:
            draw.text((x0 + 14, y), f"• {line}", font=font, fill=(240, 240, 240, 230))
            y += 18
        return img


# ----------------------------- Convenience wrappers -----------------------------

def frame_coarse_filter(rec: PCSequenceRecorder, points: np.ndarray, keep_mask: np.ndarray, title="Coarse filter (keep vs remove)"):
    notes = [
        f"Total: {points.shape[0]}",
        f"Kept: {int(keep_mask.sum())}",
        f"Removed: {int((~keep_mask).sum())}",
    ]
    rec.add_mask_view(title, points, keep_mask, notes=notes)


def frame_dbscan(rec: PCSequenceRecorder, points: np.ndarray, labels: np.ndarray, title="DBSCAN clusters"):
    n_clusters = len(np.unique(labels[labels >= 0]))
    n_noise = int(np.sum(labels < 0))
    notes = [
        f"Clusters: {n_clusters}",
        f"Noise: {n_noise}",
        "Color = cluster ID (noise in gray)",
    ]
    rec.add_labels_view(title, points, labels, notes=notes)


def frame_statistical_outlier_removal(rec: PCSequenceRecorder, points: np.ndarray, keep_mask: np.ndarray, title="Statistical Outlier Removal"):
    notes = [
        f"Kept: {int(keep_mask.sum())}",
        f"Removed: {int((~keep_mask).sum())}",
        "Red = removed as statistical outliers",
    ]
    rec.add_mask_view(title, points, keep_mask, notes=notes)


def frame_plane_fit(
    rec: PCSequenceRecorder,
    points: np.ndarray,
    inlier_mask: np.ndarray,
    plane_params: Optional[Tuple[np.ndarray, float]] = None,
    residuals_as_color: bool = False,
    title: str = "Plane fit (inliers vs outliers)",
):
    notes = [
        f"Inliers: {int(inlier_mask.sum())}",
        f"Outliers: {int((~inlier_mask).sum())}",
        "Semi-transparent surface = fitted plane",
    ]
    if residuals_as_color:
        notes.append("Color = signed distance to plane (coolwarm)")
    rec.add_plane_fit_view(
        title, points, inlier_mask,
        plane_params=plane_params,
        residuals_as_color=residuals_as_color,
        notes=notes
    )
