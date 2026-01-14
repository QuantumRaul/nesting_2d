"""
Candidate point generation for polygon placement.

Candidate points are positions where a polygon can potentially be placed.
Two strategies:
1. Perfect points: collision-free positions (for initial solution)
2. Arrangement points: intersection points of NFPs (for optimization)
"""
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import random
import numpy as np

from .geometry import get_polygon_vertices, get_bounding_box, TOLERANCE
from .nfp import translate_nfp


class CandidatePoints:
    """
    Generates candidate placement points for a polygon.

    Uses NFPs of already-placed polygons to find valid positions.
    Supports optimized rectangle mode for faster computation.
    """

    def __init__(self, max_points: int = 300, is_rect_mode: bool = False):
        """
        Initialize candidate point generator.

        Args:
            max_points: Maximum number of candidate points to return
            is_rect_mode: If True, use rectangle-optimized point generation
        """
        self.max_points = max_points
        self.is_rect_mode = is_rect_mode
        self.boundary: Optional[Polygon] = None  # IFR
        self.nfps: List[Polygon] = []
        self.translations: List[Tuple[float, float]] = []

        # Bounding box of IFR
        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0

        # Rectangle mode data
        self.rect_bounds: List[Tuple[float, float, float, float]] = []

    def set_boundary(self, ifr: Polygon) -> None:
        """
        Set the Inner-Fit Rectangle (valid placement boundary).

        Args:
            ifr: The IFR polygon
        """
        self.boundary = ifr
        if not ifr.is_empty:
            self.xmin, self.ymin, self.xmax, self.ymax = get_bounding_box(ifr)
        else:
            self.xmin = self.ymin = self.xmax = self.ymax = 0.0

    def add_nfp(self, nfp: Polygon, translate_x: float, translate_y: float) -> None:
        """
        Add an NFP with its translation (position of the stationary polygon).

        Args:
            nfp: The NFP polygon
            translate_x, translate_y: Position of the stationary polygon
        """
        self.nfps.append(nfp)
        self.translations.append((translate_x, translate_y))

    def clear(self) -> None:
        """Clear all NFPs and rectangle bounds."""
        self.nfps.clear()
        self.translations.clear()
        self.rect_bounds.clear()

    def add_rect(self, x: float, y: float, w: float, h: float) -> None:
        """
        Add a placed rectangle for rectangle mode.

        Args:
            x, y: Position of rectangle
            w, h: Dimensions of rectangle
        """
        self.rect_bounds.append((x, y, w, h))

    def is_valid_point(self, x: float, y: float) -> bool:
        """
        Check if a point is within the valid boundary (IFR).

        Args:
            x, y: Point coordinates

        Returns:
            True if point is valid
        """
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax)

    def get_perfect_points(self) -> List[Tuple[float, float]]:
        """
        Get collision-free placement points (perfect points).

        These are vertices of (IFR - union of all translated NFPs).
        Used for initial solution placement.

        Returns:
            List of (x, y) coordinates
        """
        if self.boundary is None or self.boundary.is_empty:
            return []

        if not self.nfps:
            # No obstacles, return IFR vertices
            return get_polygon_vertices(self.boundary)

        # Translate each NFP to its position
        translated_nfps = []
        for nfp, (tx, ty) in zip(self.nfps, self.translations):
            if nfp is not None and not nfp.is_empty:
                translated = translate_nfp(nfp, tx, ty)
                translated_nfps.append(translated)

        if not translated_nfps:
            return get_polygon_vertices(self.boundary)

        # Union all NFPs
        try:
            all_nfps = unary_union(translated_nfps)
        except Exception:
            all_nfps = translated_nfps[0]

        # Target region = IFR - all_nfps
        try:
            target_region = self.boundary.difference(all_nfps)
        except Exception:
            return get_polygon_vertices(self.boundary)

        if target_region.is_empty:
            return []

        # Extract vertices from target region
        points = []

        if target_region.geom_type == 'Polygon':
            points.extend(get_polygon_vertices(target_region))
            # Also add hole vertices
            for hole in target_region.interiors:
                points.extend(list(hole.coords)[:-1])

        elif target_region.geom_type == 'MultiPolygon':
            for poly in target_region.geoms:
                points.extend(get_polygon_vertices(poly))
                for hole in poly.interiors:
                    points.extend(list(hole.coords)[:-1])

        # Filter points within IFR
        valid_points = [p for p in points if self.is_valid_point(p[0], p[1])]

        # Sort by x coordinate (left-to-right placement)
        valid_points.sort(key=lambda p: (p[0], p[1]))

        return valid_points[:self.max_points]

    def get_arrangement_points(self) -> List[Tuple[float, float]]:
        """
        Get candidate points from NFP arrangement (intersections).

        These include:
        - IFR vertices
        - NFP vertices
        - Intersection points of NFP edges with IFR and each other

        Used during overlap minimization phase.

        Returns:
            List of (x, y) coordinates
        """
        if self.boundary is None or self.boundary.is_empty:
            return []

        points = []

        # Add IFR vertices
        points.extend(get_polygon_vertices(self.boundary))

        # Get IFR edges
        ifr_coords = list(self.boundary.exterior.coords)
        ifr_edges = []
        for i in range(len(ifr_coords) - 1):
            edge = LineString([ifr_coords[i], ifr_coords[i + 1]])
            ifr_edges.append(edge)

        # Process NFPs
        all_edges = ifr_edges.copy()

        for nfp, (tx, ty) in zip(self.nfps, self.translations):
            if nfp is None or nfp.is_empty:
                continue

            # Translate NFP
            translated = translate_nfp(nfp, tx, ty)

            # Add NFP vertices
            nfp_vertices = get_polygon_vertices(translated)
            for v in nfp_vertices:
                if self.is_valid_point(v[0], v[1]):
                    points.append(v)

            # Add NFP edges
            nfp_coords = list(translated.exterior.coords)
            for i in range(len(nfp_coords) - 1):
                edge = LineString([nfp_coords[i], nfp_coords[i + 1]])
                all_edges.append(edge)

        # Find intersection points between edges
        for i, edge1 in enumerate(all_edges):
            for edge2 in all_edges[i + 1:]:
                try:
                    intersection = edge1.intersection(edge2)
                    if intersection.is_empty:
                        continue

                    if intersection.geom_type == 'Point':
                        x, y = intersection.x, intersection.y
                        if self.is_valid_point(x, y):
                            points.append((x, y))

                    elif intersection.geom_type == 'MultiPoint':
                        for pt in intersection.geoms:
                            x, y = pt.x, pt.y
                            if self.is_valid_point(x, y):
                                points.append((x, y))
                except Exception:
                    continue

        # Remove duplicates (round to avoid floating point issues)
        seen = set()
        unique_points = []
        for x, y in points:
            key = (round(x, 6), round(y, 6))
            if key not in seen:
                seen.add(key)
                unique_points.append((x, y))

        # Shuffle for randomness
        random.shuffle(unique_points)

        return unique_points[:self.max_points]

    def get_random_points(self, n: int = 50) -> List[Tuple[float, float]]:
        """
        Generate random points within the IFR.

        Args:
            n: Number of points to generate

        Returns:
            List of (x, y) coordinates
        """
        if self.boundary is None or self.boundary.is_empty:
            return []

        points = []
        for _ in range(n * 2):  # Generate more, filter later
            x = random.uniform(self.xmin, self.xmax)
            y = random.uniform(self.ymin, self.ymax)
            if self.boundary.contains(Point(x, y)):
                points.append((x, y))
            if len(points) >= n:
                break

        return points

    def get_perfect_points_rect(self, new_dims: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Get collision-free placement points for rectangles.

        Uses simple corner-based candidate generation instead of
        expensive polygon boolean operations.

        Args:
            new_dims: (width, height) of rectangle to place

        Returns:
            List of (x, y) coordinates
        """
        from .rect_utils import get_rect_candidate_points

        ifr_bounds = (self.xmin, self.ymin,
                      self.xmax - self.xmin, self.ymax - self.ymin)
        return get_rect_candidate_points(ifr_bounds, self.rect_bounds, new_dims)

    def get_arrangement_points_rect(self, new_dims: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Get arrangement points for rectangle optimization.

        Uses corner-based generation with more candidate positions.

        Args:
            new_dims: (width, height) of rectangle to place

        Returns:
            List of (x, y) coordinates
        """
        from .rect_utils import get_rect_arrangement_points

        ifr_bounds = (self.xmin, self.ymin,
                      self.xmax - self.xmin, self.ymax - self.ymin)
        return get_rect_arrangement_points(ifr_bounds, self.rect_bounds, new_dims)


def generate_initial_placement_point(
    ifr: Polygon,
    nfps: List[Tuple[Polygon, float, float]]
) -> Optional[Tuple[float, float]]:
    """
    Find a collision-free placement point.

    Args:
        ifr: Inner-Fit Rectangle
        nfps: List of (NFP, translate_x, translate_y) tuples

    Returns:
        (x, y) if found, None otherwise
    """
    cp = CandidatePoints()
    cp.set_boundary(ifr)

    for nfp, tx, ty in nfps:
        cp.add_nfp(nfp, tx, ty)

    points = cp.get_perfect_points()

    if points:
        return points[0]

    return None
