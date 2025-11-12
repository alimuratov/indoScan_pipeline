"""Triangle pruning strategies for Delaunay triangulation quality control.

Strategies control which triangles are removed from a Delaunay triangulation
to eliminate outliers/noise. Use PerimeterOnlyPruningStrategy (default) for
potholes where noise appears only at edges, or GlobalPruningStrategy for
datasets with interior outliers.

Example:
    from scripts.pcdtools.strategies import GlobalPruningStrategy
    from scripts.pcdtools.pipeline import run_geometry_pipeline, VisualizationConfig

    viz = VisualizationConfig(save_hull_2d=True)
    run_geometry_pipeline(
        "pothole.pcd",
        visualization=viz,
        triangle_pruning=GlobalPruningStrategy(),  # Use global pruning
    )

For custom filter chains:
    from scripts.pcdtools.strategies.triangle_pruning import (
        PerimeterOnlyPruningStrategy, AlphaFilter
    )
    custom = PerimeterOnlyPruningStrategy(filters=[AlphaFilter()])  # Alpha only

Available Strategies:
    - PerimeterOnlyPruningStrategy: Prunes only boundary triangles (default)
    - GlobalPruningStrategy: Prunes all triangles (more aggressive)

Available Filters:
    - FlatnessFilter: Removes skinny/sliver triangles
    - AlphaFilter: Removes long-edge triangles based on percentile thresholds
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


def _default_filters() -> list["TrianglesFilter"]:
    """Default filter chain: flatness then alpha."""
    return [FlatnessFilter(), AlphaFilter()]


def _get_boundary_and_interior_tris(
    xy: np.ndarray, tris: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate boundary (convex-hull touching) from interior triangles.

    Returns:
        (boundary_tris, interior_tris) as (K, 3) and (M, 3) arrays.
    """
    import matplotlib.tri as mtri

    tri_obj = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
    neighbors = tri_obj.neighbors  # (n_tri, 3), -1 indicates hull edge
    boundary_mask = (neighbors == -1).any(axis=1)
    return tri_obj.triangles[boundary_mask], tri_obj.triangles[~boundary_mask]


class TrianglePruningStrategy(ABC):
    """Strategy interface for pruning Delaunay triangles prior to visualization/metrics."""

    # decorator to enforce implementation of the abstract method
    # you cannot instantiate TrianglePruningStrategy unless a subclass implements "prune"
    @abstractmethod
    def prune(
        self,
        xy: np.ndarray,
        tris: np.ndarray,
        *,
        edge_percentile: float,
        circ_percentile: float,
        min_circle_ratio: float,
    ) -> np.ndarray:
        """Return pruned triangle indices with shape (n, 3)."""
        raise NotImplementedError


class TrianglesFilter(ABC):
    """Composable triangle filter; can be applied globally or to a subset by the strategy."""

    @abstractmethod
    def apply(
        self,
        xy: np.ndarray,
        tris: np.ndarray,
        *,
        edge_percentile: float,
        circ_percentile: float,
        min_circle_ratio: float,
    ) -> np.ndarray:
        """Return filtered triangle indices with shape (k, 3)."""
        raise NotImplementedError


class FlatnessFilter(TrianglesFilter):
    """Remove skinny/sliver triangles using incircle-to-circumcircle ratio."""

    def apply(
        self,
        xy: np.ndarray,
        tris: np.ndarray,
        *,
        edge_percentile: float,
        circ_percentile: float,
        min_circle_ratio: float,
    ) -> np.ndarray:
        if tris.size == 0:
            return tris
        try:
            import matplotlib.tri as mtri

            tri_obj = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
            analyzer = mtri.TriAnalyzer(tri_obj)
            flat_mask = analyzer.get_flat_tri_mask(
                min_circle_ratio=min_circle_ratio)
            keep_mask = ~flat_mask
            return tri_obj.triangles[keep_mask]
        except (ImportError, ValueError, AttributeError) as e:
            # Silently fall back to original triangles if analysis fails
            return tris


class AlphaFilter(TrianglesFilter):
    """Apply alpha-like pruning to remove long/skewed triangles."""

    def apply(
        self,
        xy: np.ndarray,
        tris: np.ndarray,
        *,
        edge_percentile: float,
        circ_percentile: float,
        min_circle_ratio: float,
    ) -> np.ndarray:
        if tris.size == 0:
            return tris
        try:
            return _alpha_filter_triangles(
                xy,
                tris,
                edge_percentile=edge_percentile,
                circ_percentile=circ_percentile,
            )
        except (ValueError, IndexError) as e:
            # Silently fall back if alpha filtering fails
            return tris


def _alpha_filter_triangles(
    xy: np.ndarray,
    triangles: np.ndarray,
    *,
    edge_percentile: float = 95.0,
    circ_percentile: Optional[float] = None,
) -> np.ndarray:
    """Restrict long/skinny simplices by pruning triangles with
    - any edge length above the given edge-length percentile threshold
    - optional circumradius above the given percentile threshold

    Args:
        xy: (N,2) XY coordinates
        triangles: (M,3) int indices
        edge_percentile: e.g., 90–95 for aggressive filtering
        circ_percentile: set (e.g., 95) to also bound circumradius
    Returns:
        Filtered triangles of shape (K,3)
    """
    if triangles.size == 0:
        return triangles

    p0 = xy[triangles[:, 0]]
    p1 = xy[triangles[:, 1]]
    p2 = xy[triangles[:, 2]]
    e01 = np.linalg.norm(p1 - p0, axis=1)
    e12 = np.linalg.norm(p2 - p1, axis=1)
    e20 = np.linalg.norm(p0 - p2, axis=1)

    # Edge-length threshold from global percentile
    all_edges = np.concatenate([e01, e12, e20])
    alpha_edge = float(np.percentile(all_edges, edge_percentile))
    keep = (e01 <= alpha_edge) & (e12 <= alpha_edge) & (e20 <= alpha_edge)

    if circ_percentile is not None:
        # Circumradius R = a*b*c / (4*A), with A = triangle area in XY
        a, b, c = e01, e12, e20
        area = 0.5 * np.abs(np.cross(p1 - p0, p2 - p0))
        safe_area = np.where(area > 1e-15, area, np.inf)
        R = (a * b * c) / (4.0 * safe_area)
        finite_R = R[np.isfinite(R)]
        if finite_R.size > 0:
            alpha_circ = float(np.percentile(finite_R, circ_percentile))
            keep &= (R <= alpha_circ)

    return triangles[keep]


class PerimeterOnlyPruningStrategy(TrianglePruningStrategy):
    """Prune only boundary (convex-hull touching) triangles by flatness and alpha filters.

    Use this strategy when you suspect outliers/noise only at the perimeter of your
    pothole point cloud. Interior triangles are preserved regardless of size, ensuring
    no valid data is removed from the middle of the pothole.

    Args:
        filters: Optional list of TrianglesFilter to apply in sequence. Defaults to
            [FlatnessFilter(), AlphaFilter()].
    """

    def __init__(self, filters: Optional[list[TrianglesFilter]] = None) -> None:
        self.filters: list[TrianglesFilter] = filters if filters is not None else _default_filters(
        )

    def prune(
        self,
        xy: np.ndarray,
        tris: np.ndarray,
        *,
        edge_percentile: float,
        circ_percentile: float,
        min_circle_ratio: float,
    ) -> np.ndarray:
        if tris.size == 0:
            return tris
        try:
            current_tris = tris
            for flt in self.filters:
                boundary_tris, interior_tris = _get_boundary_and_interior_tris(
                    xy, current_tris)

                if boundary_tris.size > 0:
                    pruned_boundary_tris = flt.apply(
                        xy,
                        boundary_tris,
                        edge_percentile=edge_percentile,
                        circ_percentile=circ_percentile,
                        min_circle_ratio=min_circle_ratio,
                    )
                else:
                    pruned_boundary_tris = boundary_tris

                # Merge interior + pruned boundary
                if interior_tris.size > 0 and pruned_boundary_tris.size > 0:
                    current_tris = np.vstack(
                        [interior_tris, pruned_boundary_tris])
                elif interior_tris.size > 0:
                    current_tris = interior_tris
                else:
                    current_tris = pruned_boundary_tris

            return current_tris
        except (ImportError, ValueError, AttributeError, IndexError) as e:
            # Fall back to original triangles if boundary detection or filtering fails
            return tris


class GlobalPruningStrategy(TrianglePruningStrategy):
    """Prune all triangles (boundary and interior) using flatness and alpha filters.

    Use this strategy when outliers/noise can appear anywhere in your point cloud,
    not just at the edges. More aggressive than PerimeterOnlyPruningStrategy but
    may remove valid interior triangles if they're large or skinny.

    Args:
        filters: Optional list of TrianglesFilter to apply in sequence. Defaults to
            [FlatnessFilter(), AlphaFilter()].
    """

    def __init__(self, filters: Optional[list[TrianglesFilter]] = None) -> None:
        self.filters: list[TrianglesFilter] = filters if filters is not None else _default_filters(
        )

    def prune(
        self,
        xy: np.ndarray,
        tris: np.ndarray,
        *,
        edge_percentile: float,
        circ_percentile: float,
        min_circle_ratio: float,
    ) -> np.ndarray:
        if tris.size == 0:
            return tris
        try:
            current_tris = tris
            for flt in self.filters:
                current_tris = flt.apply(
                    xy,
                    current_tris,
                    edge_percentile=edge_percentile,
                    circ_percentile=circ_percentile,
                    min_circle_ratio=min_circle_ratio,
                )
                if current_tris.size == 0:
                    break
            return current_tris
        except (ImportError, ValueError, AttributeError, IndexError) as e:
            # Fall back to original triangles if filtering fails
            return tris
