"""
Layout class - manages the algorithm state during nesting optimization.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .data_types import Item, Sheet, TransformedShape, Solution
from .geometry import normalize_polygon, polygon_area, get_bounding_box
from .nfp import compute_nfp, compute_pd


class Layout:
    """
    Manages the state of the nesting algorithm.

    Contains:
    - Sheet and parts configuration
    - Caches for NFP and PD computations
    - Current and best solutions
    - Algorithm hyperparameters
    """

    def __init__(self, items: List[Item], sheet: Sheet):
        """
        Initialize the layout with items and sheet.

        Args:
            items: List of items to place (with quantities)
            sheet: The sheet to place items on
        """
        self.sheet = sheet
        self.original_items = items

        # Expand items by quantity and create TransformedShapes
        self.parts: List[TransformedShape] = []
        self.total_area = 0.0

        for idx, item in enumerate(items):
            # Normalize polygon (bottom-left at origin)
            normalized = normalize_polygon(item.polygon)

            for _ in range(item.quantity):
                shape = TransformedShape(
                    base=normalized,
                    item_idx=idx,
                    allowed_rotations=item.allowed_rotations,
                    rotation=0,
                    translate_x=0.0,
                    translate_y=0.0
                )
                self.parts.append(shape)
                self.total_area += polygon_area(normalized)

        self.poly_num = len(self.parts)

        # Rectangle optimization: detect if all items are rectangles
        self.is_rect_mode = self._detect_rectangle_mode()
        if self.is_rect_mode:
            self._init_rect_data()

        # Caches
        self.nfp_cache: Dict = {}
        self.pd_cache: Dict = {}

        # Penetration depth matrix (upper triangular, stored as 1D)
        # pd_matrix[i,j] maps to glob_pd[poly_num*i - i*(i+1)/2 + (j-i-1)]
        matrix_size = self.poly_num * (self.poly_num - 1) // 2
        self.glob_pd = np.zeros(matrix_size)
        self.miu = np.ones(matrix_size)  # Weights for guided search

        # Solution tracking
        self.cur_length: float = sheet.width
        self.best_length: float = float('inf')
        self.best_utilization: float = 0.0
        self.best_result: List[TransformedShape] = []
        self.lower_length: float = self.total_area / sheet.height  # Theoretical minimum

        # Algorithm hyperparameters
        self.rinc = 0.01        # 1% increase when expanding
        self.rdec = 0.04        # 4% decrease when shrinking
        self.max_iterations = 50
        self.max_pd_cache = 200000

        # Statistics
        self.pd_count = 0
        self.pd_miss = 0

    def _detect_rectangle_mode(self) -> bool:
        """Check if all items are rectangles."""
        from .rect_utils import is_rectangle
        for item in self.original_items:
            if not is_rectangle(item.polygon):
                return False
        return True

    def _init_rect_data(self) -> None:
        """Pre-compute rectangle dimensions for all parts."""
        from .rect_utils import polygon_to_rect_dims
        for part in self.parts:
            w, h = polygon_to_rect_dims(part.base)
            part.rect_width = w
            part.rect_height = h
            part.is_rect = True
        print(f"Rectangle mode enabled for {self.poly_num} parts")

    def _pd_index(self, i: int, j: int) -> int:
        """Convert (i, j) pair to 1D index for upper triangular matrix."""
        if i > j:
            i, j = j, i
        return self.poly_num * i - i * (i + 1) // 2 + (j - i - 1)

    def get_pd(self, i: int, j: int) -> float:
        """Get weighted penetration depth between shapes i and j."""
        if i == j:
            return 0.0
        idx = self._pd_index(i, j)
        return self.glob_pd[idx] * self.miu[idx]

    def get_pure_pd(self, i: int, j: int) -> float:
        """Get unweighted penetration depth between shapes i and j."""
        if i == j:
            return 0.0
        idx = self._pd_index(i, j)
        return self.glob_pd[idx]

    def set_pd(self, i: int, j: int, pd: float) -> None:
        """Set penetration depth between shapes i and j."""
        if i == j:
            return
        idx = self._pd_index(i, j)
        self.glob_pd[idx] = pd

    def get_miu(self, i: int, j: int) -> float:
        """Get weight for penetration depth between shapes i and j."""
        if i == j:
            return 0.0
        idx = self._pd_index(i, j)
        return self.miu[idx]

    def initialize_miu(self) -> None:
        """Reset all weights to 1."""
        self.miu.fill(1.0)

    def update_miu(self) -> None:
        """
        Update weights based on penetration depths.
        Increases weights for pairs with persistent overlaps.
        """
        for idx in range(len(self.glob_pd)):
            if self.glob_pd[idx] > 0:
                self.miu[idx] *= 1.1  # Increase weight by 10%

    def get_one_polygon_pd(self, p: int) -> float:
        """Get total weighted PD for polygon p with all others."""
        total = 0.0
        for i in range(self.poly_num):
            total += self.get_pd(p, i)
        return total

    def get_total_pd(self) -> float:
        """Get total weighted PD sum for all polygon pairs."""
        return float(np.sum(self.glob_pd * self.miu))

    def get_pure_total_pd(self) -> float:
        """Get total unweighted PD sum for all polygon pairs."""
        return float(np.sum(self.glob_pd))

    def update_cur_length(self) -> None:
        """Update current length based on rightmost polygon position."""
        max_x = 0.0
        for part in self.parts:
            _, _, maxx, _ = get_bounding_box(part.transformed)
            if maxx > max_x:
                max_x = maxx
        self.cur_length = max_x
        self.sheet.set_width(self.cur_length)

    def save_best_result(self) -> None:
        """Save current placement as best result."""
        self.best_result = [part.copy() for part in self.parts]
        self.best_length = self.cur_length
        self.best_utilization = self.total_area / (self.best_length * self.sheet.height)

    def get_solution(self, time_elapsed: float) -> Solution:
        """
        Create a Solution object from current best result.

        Args:
            time_elapsed: Time taken so far

        Returns:
            Solution object
        """
        return Solution(
            length=self.best_length,
            utilization=self.best_utilization,
            time=time_elapsed,
            shapes=[part.copy() for part in self.best_result]
        )

    def clear_pd_cache(self) -> None:
        """Clear the PD cache if it's getting too large."""
        if len(self.pd_cache) > self.max_pd_cache:
            self.pd_cache.clear()

    def compute_pd_for_pair(self, i: int, j: int) -> float:
        """
        Compute and store PD between shapes i and j.

        Args:
            i, j: Shape indices

        Returns:
            Penetration depth
        """
        if i == j:
            return 0.0

        # Use fast rectangle computation if in rectangle mode
        if self.is_rect_mode:
            return self.compute_pd_for_pair_rect(i, j)

        shape_i = self.parts[i]
        shape_j = self.parts[j]

        # Get NFP
        nfp = compute_nfp(
            shape_j.base, shape_j.rotation,
            shape_i.base, shape_i.rotation,
            self.nfp_cache
        )

        if nfp is None:
            pd = 0.0
        else:
            # Relative position
            rel_x = shape_i.translate_x - shape_j.translate_x
            rel_y = shape_i.translate_y - shape_j.translate_y

            # Check cache
            cache_key = (id(nfp), round(rel_x, 6), round(rel_y, 6))
            self.pd_count += 1

            if cache_key in self.pd_cache:
                pd = self.pd_cache[cache_key]
            else:
                self.pd_miss += 1
                pd = compute_pd(nfp, rel_x, rel_y)
                self.pd_cache[cache_key] = pd

        self.set_pd(i, j, pd)
        return pd

    def compute_pd_for_pair_rect(self, i: int, j: int) -> float:
        """
        Fast PD computation for rectangles using AABB overlap.

        Args:
            i, j: Shape indices

        Returns:
            Penetration depth
        """
        from .rect_utils import compute_rect_pd

        xi, yi, wi, hi = self.parts[i].get_rect_bounds()
        xj, yj, wj, hj = self.parts[j].get_rect_bounds()

        pd = compute_rect_pd(xi, yi, wi, hi, xj, yj, wj, hj)
        self.set_pd(i, j, pd)
        return pd

    def recompute_all_pd(self) -> None:
        """Recompute all penetration depths."""
        for i in range(self.poly_num):
            for j in range(i + 1, self.poly_num):
                self.compute_pd_for_pair(i, j)

    def debug_info(self) -> str:
        """Return debug information about current state."""
        return (
            f"Polygons: {self.poly_num}, "
            f"Current length: {self.cur_length:.2f}, "
            f"Best length: {self.best_length:.2f}, "
            f"Best util: {self.best_utilization:.2%}, "
            f"Total PD: {self.get_pure_total_pd():.4f}, "
            f"NFP cache: {len(self.nfp_cache)}, "
            f"PD cache: {len(self.pd_cache)}"
        )
