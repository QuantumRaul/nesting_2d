"""
No-Fit Polygon (NFP) and related geometric computations.

The NFP of two polygons A and B is the locus of points where B can be placed
such that it touches but does not overlap A. It's computed using Minkowski sum:
NFP = A (+) (-B)  (Minkowski sum of A and reflected B)
"""
from typing import Tuple, Dict, Optional, List
from shapely.geometry import Polygon, Point
from shapely import affinity
import pyclipper
import numpy as np

from .geometry import (
    TOLERANCE, rotate_polygon, translate_polygon, scale_polygon,
    get_bounding_box, point_in_polygon, distance_to_boundary
)


# Scale factor for pyclipper (works with integers)
CLIPPER_SCALE = 1e7


def _polygon_to_clipper(polygon: Polygon) -> List[Tuple[int, int]]:
    """Convert shapely polygon to pyclipper format (scaled integers)."""
    coords = list(polygon.exterior.coords)[:-1]
    return [(int(x * CLIPPER_SCALE), int(y * CLIPPER_SCALE)) for x, y in coords]


def _clipper_to_polygon(path: List[Tuple[int, int]]) -> Polygon:
    """Convert pyclipper path back to shapely polygon."""
    coords = [(x / CLIPPER_SCALE, y / CLIPPER_SCALE) for x, y in path]
    if len(coords) < 3:
        return Polygon()
    return Polygon(coords)


def compute_nfp_pyclipper(poly_a: Polygon, poly_b: Polygon) -> Optional[Polygon]:
    """
    Compute the No-Fit Polygon (NFP) using pyclipper's Minkowski sum.

    NFP = A (+) (-B)

    The NFP represents all positions where B's reference point can be placed
    such that B touches but does not overlap A.

    Args:
        poly_a: The stationary polygon A
        poly_b: The orbiting polygon B (will be reflected)

    Returns:
        The NFP as a Polygon, or None if computation fails
    """
    # Reflect B (scale by -1)
    minus_b = scale_polygon(poly_b, -1)

    # Convert to clipper format
    path_a = _polygon_to_clipper(poly_a)
    path_b = _polygon_to_clipper(minus_b)

    # Compute Minkowski sum
    try:
        result = pyclipper.MinkowskiSum(path_a, path_b, True)
        if not result or len(result) == 0:
            return None

        # Take the outer boundary (largest result)
        largest = max(result, key=lambda p: abs(pyclipper.Area(p)))
        nfp = _clipper_to_polygon(largest)

        if nfp.is_empty or not nfp.is_valid:
            return None

        return nfp

    except Exception as e:
        print(f"NFP computation error: {e}")
        return None


def compute_nfp(poly_a: Polygon, rotation_a: int,
                poly_b: Polygon, rotation_b: int,
                cache: Optional[Dict] = None) -> Optional[Polygon]:
    """
    Compute NFP with rotation support and optional caching.

    Args:
        poly_a: Stationary polygon A (base shape)
        rotation_a: Rotation index for A (0-3)
        poly_b: Orbiting polygon B (base shape)
        rotation_b: Rotation index for B (0-3)
        cache: Optional dictionary for caching results

    Returns:
        The NFP polygon, or None if computation fails
    """
    # Create cache key
    key = (id(poly_a), id(poly_b), rotation_a, rotation_b)

    # Check cache
    if cache is not None and key in cache:
        return cache[key]

    # Apply rotations
    rotated_a = rotate_polygon(poly_a, rotation_a)
    rotated_b = rotate_polygon(poly_b, rotation_b)

    # Compute NFP
    nfp = compute_nfp_pyclipper(rotated_a, rotated_b)

    # Cache result
    if cache is not None and nfp is not None:
        cache[key] = nfp

    return nfp


def compute_ifr(sheet: Polygon, part: Polygon) -> Polygon:
    """
    Compute the Inner-Fit Rectangle (IFR).

    The IFR is the region where the reference point of 'part' can be placed
    such that 'part' stays entirely within 'sheet'.

    For a rectangular sheet and convex part, this is simply a smaller rectangle.

    Args:
        sheet: The containing sheet polygon
        part: The part to be placed

    Returns:
        The IFR as a polygon (typically a rectangle)
    """
    sheet_minx, sheet_miny, sheet_maxx, sheet_maxy = get_bounding_box(sheet)
    part_minx, part_miny, part_maxx, part_maxy = get_bounding_box(part)

    # The reference point is the bottom-left corner of the part's bounding box
    # IFR = valid region for this reference point

    # Part dimensions relative to its reference point
    part_width = part_maxx - part_minx
    part_height = part_maxy - part_miny

    # Valid X range: reference point can go from (sheet_minx - part_minx) to (sheet_maxx - part_maxx)
    ifr_minx = sheet_minx - part_minx
    ifr_maxx = sheet_maxx - part_maxx
    ifr_miny = sheet_miny - part_miny
    ifr_maxy = sheet_maxy - part_maxy

    # If IFR is degenerate, return empty polygon
    if ifr_maxx < ifr_minx or ifr_maxy < ifr_miny:
        return Polygon()

    return Polygon([
        (ifr_minx, ifr_miny),
        (ifr_maxx, ifr_miny),
        (ifr_maxx, ifr_maxy),
        (ifr_minx, ifr_maxy)
    ])


def compute_pd(nfp: Polygon, x: float, y: float,
               cache: Optional[Dict] = None) -> float:
    """
    Compute the Penetration Depth (PD) for a point relative to an NFP.

    PD is the minimum distance the point must move to exit the NFP.
    If the point is outside the NFP, PD = 0.

    Args:
        nfp: The No-Fit Polygon
        x, y: The point coordinates
        cache: Optional dictionary for caching results

    Returns:
        The penetration depth (0 if point is outside NFP)
    """
    if nfp is None or nfp.is_empty:
        return 0.0

    # Create cache key (round coordinates for cache efficiency)
    if cache is not None:
        key = (id(nfp), round(x, 6), round(y, 6))
        if key in cache:
            return cache[key]

    # Quick bounding box check
    minx, miny, maxx, maxy = get_bounding_box(nfp)
    if x <= minx or x >= maxx or y <= miny or y >= maxy:
        pd = 0.0
    else:
        point = Point(x, y)

        # Check if point is inside NFP
        if nfp.contains(point):
            # PD is distance to nearest boundary
            pd = distance_to_boundary(nfp, x, y)
        else:
            pd = 0.0

    # Cache result
    if cache is not None:
        cache[key] = pd

    return pd


def compute_pd_between_shapes(shape_a, shape_b, nfp_cache: Dict,
                              pd_cache: Optional[Dict] = None) -> float:
    """
    Compute penetration depth between two TransformedShapes.

    Args:
        shape_a: First transformed shape
        shape_b: Second transformed shape
        nfp_cache: Cache for NFP computations
        pd_cache: Optional cache for PD computations

    Returns:
        Penetration depth (0 if no overlap)
    """
    # Get NFP with shape_a stationary, shape_b orbiting
    nfp = compute_nfp(
        shape_a.base, shape_a.rotation,
        shape_b.base, shape_b.rotation,
        nfp_cache
    )

    if nfp is None:
        return 0.0

    # Relative position: where is shape_b's reference relative to shape_a?
    rel_x = shape_b.translate_x - shape_a.translate_x
    rel_y = shape_b.translate_y - shape_a.translate_y

    return compute_pd(nfp, rel_x, rel_y, pd_cache)


def get_nfp_bounding_box(nfp: Polygon) -> Tuple[float, float, float, float]:
    """
    Get the bounding box of an NFP.

    Args:
        nfp: The NFP polygon

    Returns:
        (minx, miny, maxx, maxy)
    """
    if nfp is None or nfp.is_empty:
        return (0, 0, 0, 0)
    return get_bounding_box(nfp)


def translate_nfp(nfp: Polygon, dx: float, dy: float) -> Polygon:
    """
    Translate an NFP by (dx, dy).

    This is used when placing NFPs relative to already-placed shapes.

    Args:
        nfp: The NFP polygon
        dx, dy: Translation offsets

    Returns:
        Translated NFP
    """
    return translate_polygon(nfp, dx, dy)
