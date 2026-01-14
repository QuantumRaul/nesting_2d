"""
GOMH Algorithm - Guided Overlap Minimization Heuristic for 2D nesting.

Main algorithm for solving the irregular strip packing problem.
Includes optimized paths for rectangular shapes.
"""
import time
import random
from typing import Callable, Optional, List, Tuple
from threading import Event

from .data_types import Solution
from .layout import Layout
from .nfp import compute_nfp, compute_ifr, compute_pd
from .candidate_points import CandidatePoints
from .geometry import get_bounding_box, rotate_polygon, TOLERANCE
from .rect_utils import (
    rect_ifr, get_rect_candidate_points, get_rect_arrangement_points,
    is_collision_free_rect, compute_rect_pd, get_rotated_dims
)


def get_initial_solution(layout: Layout) -> None:
    """
    Generate an initial feasible solution.

    Places polygons one by one at collision-free "perfect points".
    Uses a greedy bottom-left strategy.
    Dispatches to rectangle-optimized version if in rectangle mode.

    Args:
        layout: The Layout object to modify
    """
    if layout.is_rect_mode:
        _get_initial_solution_rect(layout)
    else:
        _get_initial_solution_polygon(layout)


def _get_initial_solution_polygon(layout: Layout) -> None:
    """Original polygon-based initial solution (for non-rectangular shapes)."""
    num_poly = layout.poly_num
    print(f"Generating initial solution for {num_poly} polygons...")

    for i in range(num_poly):
        shape_i = layout.parts[i]
        rotated_i = shape_i.rotated_base
        rotation_i = shape_i.rotation

        # Compute IFR (Inner-Fit Rectangle)
        ifr = compute_ifr(layout.sheet.polygon, rotated_i)

        if ifr.is_empty:
            raise RuntimeError(f"Part {i} is too large for the sheet!")

        # Create candidate points generator
        cp = CandidatePoints()
        cp.set_boundary(ifr)

        # Add NFPs for all previously placed polygons
        for j in range(i):
            shape_j = layout.parts[j]

            # Compute NFP (j stationary, i orbiting)
            nfp = compute_nfp(
                shape_j.base, shape_j.rotation,
                shape_i.base, rotation_i,
                layout.nfp_cache
            )

            if nfp is not None:
                cp.add_nfp(nfp, shape_j.translate_x, shape_j.translate_y)

        # Get perfect (collision-free) points
        points = cp.get_perfect_points()

        if not points:
            raise RuntimeError(f"No valid placement found for part {i}!")

        # Place at first valid point
        x, y = points[0]
        shape_i.set_position(x, y)

        # Update PD for all previous polygons
        for j in range(i):
            pd = layout.compute_pd_for_pair(i, j)

    # Verify solution is feasible
    total_pd = layout.get_pure_total_pd()
    if total_pd > TOLERANCE:
        print(f"Warning: Initial solution has overlap (PD = {total_pd})")

    # Update current length
    layout.update_cur_length()
    layout.save_best_result()

    print(f"Initial solution: length = {layout.cur_length:.2f}, "
          f"utilization = {layout.best_utilization:.2%}")


def _get_initial_solution_rect(layout: Layout) -> None:
    """
    Fast initial solution for rectangles using bottom-left-fill.

    Uses simple AABB collision detection instead of NFP computation.
    """
    num_poly = layout.poly_num
    print(f"Generating initial solution for {num_poly} rectangles (optimized)...")

    for i in range(num_poly):
        shape_i = layout.parts[i]
        wi, hi = shape_i.get_current_dims()

        # Fast IFR calculation for rectangles
        ifr = rect_ifr(layout.sheet.width, layout.sheet.height, wi, hi)

        if ifr[2] < 0 or ifr[3] < 0:
            raise RuntimeError(f"Part {i} is too large for the sheet!")

        # Get placed rectangles
        placed = [layout.parts[j].get_rect_bounds() for j in range(i)]

        # Get candidate points (corners of placed rectangles)
        points = get_rect_candidate_points(ifr, placed, (wi, hi))

        if not points:
            raise RuntimeError(f"No valid placement found for part {i}!")

        # Find first collision-free point
        placed_found = False
        for x, y in points:
            if is_collision_free_rect(x, y, wi, hi, placed):
                shape_i.set_position(x, y)
                placed_found = True
                break

        if not placed_found:
            # Fallback: try origin
            shape_i.set_position(0, 0)

        # Update PD matrix
        for j in range(i):
            layout.compute_pd_for_pair_rect(i, j)

    # Verify solution is feasible
    total_pd = layout.get_pure_total_pd()
    if total_pd > TOLERANCE:
        print(f"Warning: Initial solution has overlap (PD = {total_pd})")

    # Update current length
    layout.update_cur_length()
    layout.save_best_result()

    print(f"Initial solution: length = {layout.cur_length:.2f}, "
          f"utilization = {layout.best_utilization:.2%}")


def shrink(layout: Layout) -> None:
    """
    Shrink the sheet length after finding a feasible solution.

    Reduces length by rdec (default 4%) and repositions
    polygons that extend beyond the new length.

    Args:
        layout: The Layout object to modify
    """
    # Calculate new length
    new_length = layout.best_length * (1 - layout.rdec)
    if new_length < layout.lower_length:
        new_length = layout.lower_length

    layout.cur_length = new_length
    layout.sheet.set_width(new_length)

    # Find polygons that extend beyond new length
    polygons_to_reposition = []
    for i, part in enumerate(layout.parts):
        _, _, maxx, _ = get_bounding_box(part.transformed)
        if maxx > new_length:
            polygons_to_reposition.append(i)

    # Reposition these polygons randomly
    for i in polygons_to_reposition:
        part = layout.parts[i]
        rotated = part.rotated_base
        ifr = compute_ifr(layout.sheet.polygon, rotated)

        if not ifr.is_empty:
            minx, miny, maxx, maxy = get_bounding_box(ifr)
            # Random position within IFR (prefer right side)
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            part.set_position(x, y)

    # Update PD for repositioned polygons
    for i in polygons_to_reposition:
        for j in range(layout.poly_num):
            if i != j:
                layout.compute_pd_for_pair(i, j)


def expand(layout: Layout) -> None:
    """
    Expand the sheet length when unable to find a feasible solution.

    Increases length by rinc (default 1%).

    Args:
        layout: The Layout object to modify
    """
    layout.cur_length *= (1 + layout.rinc)
    layout.sheet.set_width(layout.cur_length)


def minimize_overlap(layout: Layout, stop_flag: Event) -> bool:
    """
    Iteratively minimize overlaps to find a feasible solution.

    For each polygon, tries different rotations and positions
    to minimize the weighted penetration depth sum.
    Dispatches to rectangle-optimized version if in rectangle mode.

    Args:
        layout: The Layout object to modify
        stop_flag: Event to signal early termination

    Returns:
        True if a feasible solution was found
    """
    if layout.is_rect_mode:
        return _minimize_overlap_rect(layout, stop_flag)
    else:
        return _minimize_overlap_polygon(layout, stop_flag)


def _minimize_overlap_polygon(layout: Layout, stop_flag: Event) -> bool:
    """Original polygon-based overlap minimization."""
    num_iterations = 0
    min_overlap = float('inf')

    # Reset weights
    layout.initialize_miu()

    # Create index list for random ordering
    indices = list(range(layout.poly_num))

    while num_iterations < layout.max_iterations:
        if stop_flag.is_set():
            return False

        # Shuffle polygon order
        random.shuffle(indices)

        # Try to improve each polygon's position
        for idx in indices:
            if stop_flag.is_set():
                return False

            shape = layout.parts[idx]

            # Current weighted PD for this polygon
            cur_pd = layout.get_one_polygon_pd(idx)

            # Skip if no overlap
            if abs(cur_pd) < TOLERANCE:
                continue

            # Try each allowed rotation
            for rotation in shape.get_reduced_rotations():
                rotated = rotate_polygon(shape.base, rotation)

                # Compute IFR for this rotation
                ifr = compute_ifr(layout.sheet.polygon, rotated)
                if ifr.is_empty:
                    continue

                # Generate candidate points
                cp = CandidatePoints()
                cp.set_boundary(ifr)

                # Add NFPs for all other polygons
                for k in range(layout.poly_num):
                    if k == idx:
                        continue

                    shape_k = layout.parts[k]
                    nfp = compute_nfp(
                        shape_k.base, shape_k.rotation,
                        shape.base, rotation,
                        layout.nfp_cache
                    )

                    if nfp is not None:
                        cp.add_nfp(nfp, shape_k.translate_x, shape_k.translate_y)

                # Get candidate positions
                points = cp.get_arrangement_points()

                # Try each candidate point
                for x, y in points:
                    # Calculate new weighted PD at this position
                    new_pd = 0.0
                    temp_pds = []

                    for k in range(layout.poly_num):
                        if k == idx:
                            temp_pds.append(0.0)
                            continue

                        shape_k = layout.parts[k]

                        # Get NFP
                        nfp = compute_nfp(
                            shape_k.base, shape_k.rotation,
                            shape.base, rotation,
                            layout.nfp_cache
                        )

                        if nfp is None:
                            pd = 0.0
                        else:
                            rel_x = x - shape_k.translate_x
                            rel_y = y - shape_k.translate_y
                            pd = compute_pd(nfp, rel_x, rel_y, layout.pd_cache)

                        temp_pds.append(pd)
                        new_pd += pd * layout.get_miu(idx, k)

                    # If this position is better, use it
                    if new_pd < cur_pd:
                        shape.set(rotation, x, y)

                        # Update PD matrix
                        for k in range(layout.poly_num):
                            if k != idx:
                                layout.set_pd(idx, k, temp_pds[k])

                        cur_pd = new_pd

                    # If no overlap for this polygon, stop searching
                    if cur_pd < TOLERANCE:
                        break

                # Break rotation loop if found zero overlap
                if cur_pd < TOLERANCE:
                    break

        # Check total overlap
        pure_overlap = layout.get_pure_total_pd()

        if pure_overlap < TOLERANCE:
            print("Feasible solution found!")
            return True

        if pure_overlap < min_overlap:
            print(f"  Iteration {num_iterations}: overlap = {pure_overlap:.6f}")
            min_overlap = pure_overlap
            num_iterations = 0
        else:
            num_iterations += 1

        # Update weights
        layout.update_miu()

        # Clear PD cache if too large
        layout.clear_pd_cache()

    return False


def _minimize_overlap_rect(layout: Layout, stop_flag: Event) -> bool:
    """
    Fast overlap minimization for rectangles.

    Uses direct AABB overlap computation instead of NFP-based PD.
    Only tests 2 rotations (0° and 90°) for non-square rectangles.
    """
    num_iterations = 0
    min_overlap = float('inf')

    # Reset weights
    layout.initialize_miu()

    # Create index list for random ordering
    indices = list(range(layout.poly_num))

    while num_iterations < layout.max_iterations:
        if stop_flag.is_set():
            return False

        # Shuffle polygon order
        random.shuffle(indices)

        # Try to improve each rectangle's position
        for idx in indices:
            if stop_flag.is_set():
                return False

            shape = layout.parts[idx]

            # Current weighted PD for this rectangle
            cur_pd = layout.get_one_polygon_pd(idx)

            # Skip if no overlap
            if abs(cur_pd) < TOLERANCE:
                continue

            # Get other rectangles' bounds
            other_rects = []
            for k in range(layout.poly_num):
                if k != idx:
                    other_rects.append((k, layout.parts[k].get_rect_bounds()))

            # Try each allowed rotation (only 0° and 90° for rectangles)
            for rotation in shape.get_reduced_rotations():
                # Get dimensions for this rotation
                w, h = get_rotated_dims(shape.rect_width, shape.rect_height, rotation)

                # Compute IFR for this rotation
                ifr = rect_ifr(layout.sheet.width, layout.sheet.height, w, h)
                if ifr[2] < 0 or ifr[3] < 0:
                    continue

                # Get placed rectangles (excluding current)
                placed = [bounds for k, bounds in other_rects]

                # Get candidate positions
                points = get_rect_arrangement_points(ifr, placed, (w, h))

                # Try each candidate point
                for x, y in points:
                    # Calculate new weighted PD at this position
                    new_pd = 0.0
                    temp_pds = []

                    for k in range(layout.poly_num):
                        if k == idx:
                            temp_pds.append(0.0)
                            continue

                        # Fast AABB overlap computation
                        xk, yk, wk, hk = layout.parts[k].get_rect_bounds()
                        pd = compute_rect_pd(x, y, w, h, xk, yk, wk, hk)

                        temp_pds.append(pd)
                        new_pd += pd * layout.get_miu(idx, k)

                    # If this position is better, use it
                    if new_pd < cur_pd:
                        shape.set(rotation, x, y)

                        # Update PD matrix
                        for k in range(layout.poly_num):
                            if k != idx:
                                layout.set_pd(idx, k, temp_pds[k])

                        cur_pd = new_pd

                    # If no overlap for this rectangle, stop searching
                    if cur_pd < TOLERANCE:
                        break

                # Break rotation loop if found zero overlap
                if cur_pd < TOLERANCE:
                    break

        # Check total overlap
        pure_overlap = layout.get_pure_total_pd()

        if pure_overlap < TOLERANCE:
            print("Feasible solution found!")
            return True

        if pure_overlap < min_overlap:
            print(f"  Iteration {num_iterations}: overlap = {pure_overlap:.6f}")
            min_overlap = pure_overlap
            num_iterations = 0
        else:
            num_iterations += 1

        # Update weights
        layout.update_miu()

    return False


def gomh(layout: Layout, max_time: int,
         progress_callback: Callable[[Solution], None],
         stop_flag: Event) -> None:
    """
    Main GOMH algorithm for 2D nesting.

    Iteratively shrinks and expands the sheet, using minimize_overlap
    to find feasible solutions at each length.

    Args:
        layout: The Layout object containing items and sheet
        max_time: Maximum time in seconds
        progress_callback: Called when a better solution is found
        stop_flag: Event to signal early termination
    """
    start_time = time.time()

    # Generate initial solution
    try:
        get_initial_solution(layout)
    except RuntimeError as e:
        print(f"Failed to find initial solution: {e}")
        return

    # Report initial solution
    elapsed = time.time() - start_time
    progress_callback(layout.get_solution(elapsed))

    # Shrink and try to improve
    shrink(layout)

    # Track consecutive iterations without improvement to detect stagnation
    no_improvement_count = 0
    last_best_length = layout.best_length

    while (time.time() - start_time) < max_time:
        if stop_flag.is_set():
            print("Algorithm stopped by user")
            break

        print(f"\nTrying length = {layout.cur_length:.2f}")

        # Try to find feasible solution at current length
        feasible = minimize_overlap(layout, stop_flag)

        if feasible:
            # Save as best result
            layout.update_cur_length()
            old_best = layout.best_length
            layout.best_length = min(layout.best_length, layout.cur_length)
            layout.best_utilization = layout.total_area / (layout.best_length * layout.sheet.height)
            layout.best_result = [part.copy() for part in layout.parts]

            # Check if we actually improved
            if layout.best_length < old_best - 0.001:
                no_improvement_count = 0
                # Report progress
                elapsed = time.time() - start_time
                progress_callback(layout.get_solution(elapsed))

                print(f"New best: length = {layout.best_length:.2f}, "
                      f"utilization = {layout.best_utilization:.2%}")
            else:
                no_improvement_count += 1
                print(f"No improvement (best still {layout.best_length:.2f})")

            # Check if we've reached the lower bound
            if layout.best_length <= layout.lower_length * 1.001:
                print("Reached lower bound!")
                break

            # Check for stagnation
            if no_improvement_count >= 5:
                print("No improvement after 5 iterations, stopping.")
                break

            # Shrink further
            shrink(layout)
        else:
            # Couldn't find feasible solution, try expanding
            new_length = layout.cur_length * (1 + layout.rinc)
            if new_length >= layout.best_length:
                # Can't expand beyond best, and shrink would give same length
                # Try a smaller shrink to explore different lengths
                target_length = (layout.cur_length + layout.best_length) / 2
                if target_length >= layout.best_length - 0.01:
                    # Already very close to best, we're done optimizing
                    print("Cannot improve further, stopping optimization.")
                    break
                layout.cur_length = target_length
                layout.sheet.set_width(target_length)
            else:
                expand(layout)

    elapsed = time.time() - start_time
    print(f"\nAlgorithm finished in {elapsed:.1f}s")
    print(f"Best length: {layout.best_length:.2f}")
    print(f"Best utilization: {layout.best_utilization:.2%}")
