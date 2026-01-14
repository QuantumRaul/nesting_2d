"""
Data types for the 2D nesting algorithm.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from shapely.geometry import Polygon
from shapely import affinity
import copy


@dataclass
class Item:
    """
    Represents an input item (part) to be nested.

    Attributes:
        polygon: The shapely Polygon geometry
        quantity: Number of copies of this item
        allowed_rotations: Number of allowed rotations (1, 2, or 4)
            1 = no rotation (0° only)
            2 = 0° and 180°
            4 = 0°, 90°, 180°, 270°
    """
    polygon: Polygon
    quantity: int = 1
    allowed_rotations: int = 1

    @property
    def area(self) -> float:
        """Calculate the area of the polygon."""
        return abs(self.polygon.area)


@dataclass
class Sheet:
    """
    Represents the sheet/bin where items are placed.

    Attributes:
        width: Width of the sheet (can change during algorithm)
        height: Height of the sheet (fixed)
    """
    width: float
    height: float

    @property
    def polygon(self) -> Polygon:
        """Get the sheet as a rectangular Polygon."""
        return Polygon([
            (0, 0),
            (self.width, 0),
            (self.width, self.height),
            (0, self.height)
        ])

    def set_width(self, new_width: float) -> None:
        """Update the sheet width."""
        self.width = new_width


@dataclass
class TransformedShape:
    """
    Represents a placed shape with its transformation (rotation + translation).

    Attributes:
        base: Reference to the base polygon (at origin, no rotation)
        item_idx: Index of the original item
        allowed_rotations: Number of allowed rotations
        rotation: Current rotation index (0, 1, 2, or 3)
        translate_x: X translation
        translate_y: Y translation
        rect_width: Width of rectangle (if is_rect=True)
        rect_height: Height of rectangle (if is_rect=True)
        is_rect: True if this shape is a rectangle (enables optimizations)
    """
    base: Polygon
    item_idx: int
    allowed_rotations: int = 1
    rotation: int = 0
    translate_x: float = 0.0
    translate_y: float = 0.0
    _transformed: Optional[Polygon] = field(default=None, repr=False)
    # Rectangle optimization fields
    rect_width: float = 0.0
    rect_height: float = 0.0
    is_rect: bool = False

    @property
    def transformed(self) -> Polygon:
        """Get the transformed polygon (rotated and translated)."""
        if self._transformed is None:
            self._update_transformed()
        return self._transformed

    def _update_transformed(self) -> None:
        """Recompute the transformed polygon."""
        # Apply rotation first (around origin)
        angle = self.rotation * 90  # 0, 90, 180, or 270 degrees
        rotated = affinity.rotate(self.base, angle, origin=(0, 0))
        # Normalize rotated polygon so its bounding box starts at (0, 0)
        minx, miny, _, _ = rotated.bounds
        rotated = affinity.translate(rotated, -minx, -miny)
        # Then apply the final translation
        self._transformed = affinity.translate(rotated, self.translate_x, self.translate_y)

    def set_position(self, x: float, y: float) -> None:
        """Set the translation position."""
        self.translate_x = x
        self.translate_y = y
        self._transformed = None  # Invalidate cache

    def set_rotation(self, rotation: int) -> None:
        """Set the rotation index (0, 1, 2, or 3)."""
        self.rotation = rotation % self.allowed_rotations
        self._transformed = None  # Invalidate cache

    def set(self, rotation: int, x: float, y: float) -> None:
        """Set both rotation and position."""
        self.rotation = rotation % self.allowed_rotations
        self.translate_x = x
        self.translate_y = y
        self._transformed = None  # Invalidate cache

    @property
    def rotated_base(self) -> Polygon:
        """Get the base polygon with current rotation applied (no translation)."""
        angle = self.rotation * 90
        rotated = affinity.rotate(self.base, angle, origin=(0, 0))
        # Normalize so bounding box starts at (0, 0)
        minx, miny, _, _ = rotated.bounds
        return affinity.translate(rotated, -minx, -miny)

    def get_reduced_rotations(self) -> List[int]:
        """
        Get list of unique rotations (removing symmetric duplicates).
        For symmetric polygons, some rotations produce the same shape.
        For rectangles: only 0 and 1 (90 degrees) matter.
        """
        if self.is_rect:
            # For rectangles, 180° = 0° and 270° = 90°
            if abs(self.rect_width - self.rect_height) < 1e-9:
                # Square: all rotations are equivalent
                return [0]
            # Non-square rectangle: only 0° and 90° are unique
            return [0, 1] if self.allowed_rotations >= 2 else [0]
        rotations = list(range(self.allowed_rotations))
        return rotations

    def get_current_dims(self) -> Tuple[float, float]:
        """
        Get (width, height) after current rotation.
        Only valid if is_rect=True.
        """
        if self.rotation % 2 == 0:
            return (self.rect_width, self.rect_height)
        else:
            return (self.rect_height, self.rect_width)

    def get_rect_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get (x, y, width, height) of current placement.
        Only valid if is_rect=True.
        """
        w, h = self.get_current_dims()
        return (self.translate_x, self.translate_y, w, h)

    def copy(self) -> 'TransformedShape':
        """Create a deep copy of this shape."""
        return TransformedShape(
            base=self.base,
            item_idx=self.item_idx,
            allowed_rotations=self.allowed_rotations,
            rotation=self.rotation,
            translate_x=self.translate_x,
            translate_y=self.translate_y,
            rect_width=self.rect_width,
            rect_height=self.rect_height,
            is_rect=self.is_rect
        )


@dataclass
class Solution:
    """
    Represents a solution found by the algorithm.

    Attributes:
        length: The sheet length used
        utilization: Material utilization percentage (0-1)
        time: Time taken to find this solution (seconds)
        shapes: List of placed shapes
    """
    length: float
    utilization: float
    time: float
    shapes: List[TransformedShape]

    def copy(self) -> 'Solution':
        """Create a deep copy of this solution."""
        return Solution(
            length=self.length,
            utilization=self.utilization,
            time=self.time,
            shapes=[s.copy() for s in self.shapes]
        )
