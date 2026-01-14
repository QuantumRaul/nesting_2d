"""
CSV file reader for polygon data.

Reads the ESICUP benchmark CSV format:
- Column 'polygon': vertex count followed by x,y coordinates
  Format: "n x1 y1 x2 y2 ... xn yn"
- Column 'allowed_rotations': 1, 2, or 4
- Column 'quantity': number of copies
"""
import pandas as pd
from typing import List, Tuple
from shapely.geometry import Polygon

from core.data_types import Item


def parse_polygon_string(polygon_str: str) -> Polygon:
    """
    Parse a polygon string from CSV format.

    Format: "n x1 y1 x2 y2 ... xn yn"
    where n is the number of vertices.

    Args:
        polygon_str: The polygon string

    Returns:
        Shapely Polygon object
    """
    parts = polygon_str.strip().split()
    n_vertices = int(parts[0])

    coords = []
    for i in range(n_vertices):
        x = float(parts[1 + i * 2])
        y = float(parts[2 + i * 2])
        coords.append((x, y))

    return Polygon(coords)


def read_csv(filepath: str, require_rectangles: bool = False) -> List[Item]:
    """
    Read items from a CSV file.

    Args:
        filepath: Path to the CSV file
        require_rectangles: If True, validates all shapes are rectangles

    Returns:
        List of Item objects

    Raises:
        ValueError: If require_rectangles=True and non-rectangular shape found
    """
    from core.rect_utils import is_rectangle

    df = pd.read_csv(filepath)

    items = []
    for idx, row in df.iterrows():
        polygon = parse_polygon_string(str(row['polygon']))
        allowed_rotations = int(row['allowed_rotations'])
        quantity = int(row['quantity'])

        # Validate rectangle if required
        if require_rectangles and not is_rectangle(polygon):
            raise ValueError(f"Non-rectangular shape found at row {idx}")

        # For rectangles, limit to 2 meaningful rotations (0° and 90°)
        # since 180° = 0° and 270° = 90° for rectangles
        if is_rectangle(polygon) and allowed_rotations == 4:
            allowed_rotations = 2

        item = Item(
            polygon=polygon,
            quantity=quantity,
            allowed_rotations=allowed_rotations
        )
        items.append(item)

    return items


def create_sample_items() -> List[Item]:
    """
    Create sample items for testing.

    Returns:
        List of simple test items
    """
    items = []

    # Square
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    items.append(Item(polygon=square, quantity=2, allowed_rotations=4))

    # Rectangle
    rect = Polygon([(0, 0), (3, 0), (3, 1.5), (0, 1.5)])
    items.append(Item(polygon=rect, quantity=2, allowed_rotations=4))

    # Triangle
    tri = Polygon([(0, 0), (2, 0), (1, 2)])
    items.append(Item(polygon=tri, quantity=2, allowed_rotations=4))

    # L-shape
    l_shape = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 3), (0, 3)])
    items.append(Item(polygon=l_shape, quantity=2, allowed_rotations=4))

    return items


def get_total_area(items: List[Item]) -> float:
    """
    Calculate total area of all items (including quantities).

    Args:
        items: List of items

    Returns:
        Total area
    """
    total = 0.0
    for item in items:
        total += abs(item.polygon.area) * item.quantity
    return total


def get_bounding_dimensions(items: List[Item]) -> Tuple[float, float]:
    """
    Get maximum dimensions needed to fit the largest item.

    Args:
        items: List of items

    Returns:
        (max_width, max_height) considering all rotations
    """
    max_dim = 0.0

    for item in items:
        minx, miny, maxx, maxy = item.polygon.bounds
        width = maxx - minx
        height = maxy - miny
        max_dim = max(max_dim, width, height)

    return max_dim, max_dim
