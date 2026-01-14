"""
Test script for rectangle-optimized nesting.

Tests that the rectangle optimizations work correctly and produce
valid solutions for rectangular shapes.
"""
import sys
import os
import time
from threading import Event

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_types import Item, Sheet
from core.layout import Layout
from core.algorithm import gomh, get_initial_solution
from core.rect_utils import (
    is_rectangle, polygon_to_rect_dims, compute_rect_nfp,
    compute_rect_pd, rect_ifr, is_collision_free_rect,
    get_rect_candidate_points
)
from file_io.csv_reader import read_csv
from shapely.geometry import Polygon


def test_rect_utils():
    """Test rectangle utility functions."""
    print("=" * 60)
    print("Testing rect_utils functions...")
    print("=" * 60)

    # Test is_rectangle
    rect = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
    triangle = Polygon([(0, 0), (10, 0), (5, 5)])

    assert is_rectangle(rect), "Rectangle should be detected as rectangle"
    assert not is_rectangle(triangle), "Triangle should not be detected as rectangle"
    print("  is_rectangle: PASSED")

    # Test polygon_to_rect_dims
    dims = polygon_to_rect_dims(rect)
    assert dims == (10.0, 5.0), f"Expected (10.0, 5.0), got {dims}"
    print("  polygon_to_rect_dims: PASSED")

    # Test compute_rect_nfp
    nfp = compute_rect_nfp(10, 5, 4, 3)
    assert nfp == (-4, -3, 14, 8), f"Expected (-4, -3, 14, 8), got {nfp}"
    print("  compute_rect_nfp: PASSED")

    # Test compute_rect_pd (overlapping)
    pd = compute_rect_pd(0, 0, 10, 5, 8, 3, 4, 3)
    assert pd > 0, f"Expected positive PD for overlapping rectangles, got {pd}"
    print(f"  compute_rect_pd (overlapping): PASSED (pd={pd})")

    # Test compute_rect_pd (non-overlapping)
    pd = compute_rect_pd(0, 0, 10, 5, 15, 0, 4, 3)
    assert pd == 0, f"Expected 0 PD for non-overlapping rectangles, got {pd}"
    print("  compute_rect_pd (non-overlapping): PASSED")

    # Test rect_ifr
    ifr = rect_ifr(100, 50, 10, 5)
    assert ifr == (0.0, 0.0, 90.0, 45.0), f"Expected (0.0, 0.0, 90.0, 45.0), got {ifr}"
    print("  rect_ifr: PASSED")

    # Test is_collision_free_rect
    placed = [(0, 0, 10, 5), (15, 0, 8, 4)]
    assert is_collision_free_rect(25, 0, 5, 5, placed), "Should be collision-free"
    assert not is_collision_free_rect(5, 2, 5, 5, placed), "Should have collision"
    print("  is_collision_free_rect: PASSED")

    # Test get_rect_candidate_points
    ifr = (0, 0, 90, 45)
    placed = [(0, 0, 10, 5)]
    points = get_rect_candidate_points(ifr, placed, (5, 3))
    assert len(points) > 0, "Should generate candidate points"
    print(f"  get_rect_candidate_points: PASSED ({len(points)} points generated)")

    print("\nAll rect_utils tests PASSED!")
    return True


def test_layout_rect_mode():
    """Test that Layout correctly detects rectangle mode."""
    print("\n" + "=" * 60)
    print("Testing Layout rectangle mode detection...")
    print("=" * 60)

    # Create rectangle items
    rect1 = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
    rect2 = Polygon([(0, 0), (8, 0), (8, 4), (0, 4)])

    items = [
        Item(polygon=rect1, quantity=2, allowed_rotations=4),
        Item(polygon=rect2, quantity=3, allowed_rotations=4)
    ]

    sheet = Sheet(width=100, height=50)
    layout = Layout(items, sheet)

    assert layout.is_rect_mode, "Layout should detect rectangle mode"
    print(f"  Rectangle mode detected: {layout.is_rect_mode}")

    # Check that parts have rectangle data
    for i, part in enumerate(layout.parts):
        assert part.is_rect, f"Part {i} should be marked as rectangle"
        assert part.rect_width > 0, f"Part {i} should have positive rect_width"
        assert part.rect_height > 0, f"Part {i} should have positive rect_height"
        print(f"  Part {i}: {part.rect_width}x{part.rect_height}")

    print("\nLayout rectangle mode test PASSED!")
    return True


def test_initial_solution():
    """Test initial solution generation for rectangles."""
    print("\n" + "=" * 60)
    print("Testing initial solution generation...")
    print("=" * 60)

    # Create rectangle items
    items = [
        Item(polygon=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]), quantity=2, allowed_rotations=4),
        Item(polygon=Polygon([(0, 0), (8, 0), (8, 4), (0, 4)]), quantity=2, allowed_rotations=4),
        Item(polygon=Polygon([(0, 0), (6, 0), (6, 3), (0, 3)]), quantity=3, allowed_rotations=4),
    ]

    sheet = Sheet(width=200, height=50)
    layout = Layout(items, sheet)

    print(f"  Total parts: {layout.poly_num}")
    print(f"  Rectangle mode: {layout.is_rect_mode}")

    # Generate initial solution
    start = time.time()
    get_initial_solution(layout)
    elapsed = time.time() - start

    print(f"  Initial solution generated in {elapsed:.4f}s")
    print(f"  Length: {layout.cur_length:.2f}")
    print(f"  Utilization: {layout.best_utilization:.2%}")

    # Verify no overlaps (PD should be 0)
    total_pd = layout.get_pure_total_pd()
    print(f"  Total PD: {total_pd:.6f}")

    if total_pd < 1e-6:
        print("\nInitial solution test PASSED!")
        return True
    else:
        print("\nInitial solution test FAILED - overlaps detected!")
        return False


def test_full_algorithm():
    """Test full GOMH algorithm with rectangles."""
    print("\n" + "=" * 60)
    print("Testing full GOMH algorithm with rectangles...")
    print("=" * 60)

    # Create rectangle items
    items = [
        Item(polygon=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]), quantity=3, allowed_rotations=4),
        Item(polygon=Polygon([(0, 0), (8, 0), (8, 4), (0, 4)]), quantity=4, allowed_rotations=4),
        Item(polygon=Polygon([(0, 0), (6, 0), (6, 3), (0, 3)]), quantity=5, allowed_rotations=4),
        Item(polygon=Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]), quantity=2, allowed_rotations=4),
    ]

    sheet = Sheet(width=300, height=30)
    layout = Layout(items, sheet)

    print(f"  Total parts: {layout.poly_num}")
    print(f"  Total area: {layout.total_area:.2f}")
    print(f"  Rectangle mode: {layout.is_rect_mode}")

    # Track solutions
    solutions = []
    def progress_callback(solution):
        solutions.append(solution)
        print(f"  Progress: length={solution.length:.2f}, util={solution.utilization:.2%}")

    stop_flag = Event()

    # Run algorithm for short time
    start = time.time()
    gomh(layout, max_time=10, progress_callback=progress_callback, stop_flag=stop_flag)
    elapsed = time.time() - start

    print(f"\n  Algorithm completed in {elapsed:.2f}s")
    print(f"  Solutions found: {len(solutions)}")

    if solutions:
        best = solutions[-1]
        print(f"  Best length: {best.length:.2f}")
        print(f"  Best utilization: {best.utilization:.2%}")
        print("\nFull algorithm test PASSED!")
        return True
    else:
        print("\nFull algorithm test FAILED - no solutions found!")
        return False


def test_csv_loading():
    """Test loading rectangles from CSV."""
    print("\n" + "=" * 60)
    print("Testing CSV loading with rectangle validation...")
    print("=" * 60)

    csv_path = os.path.join(os.path.dirname(__file__),
                           'data', 'csv', 'rectangles_test.csv')

    if not os.path.exists(csv_path):
        print(f"  CSV file not found: {csv_path}")
        print("  Skipping CSV test")
        return True

    items = read_csv(csv_path, require_rectangles=True)
    print(f"  Loaded {len(items)} item types from CSV")

    total_parts = sum(item.quantity for item in items)
    print(f"  Total parts: {total_parts}")

    # Verify all are rectangles
    for i, item in enumerate(items):
        assert is_rectangle(item.polygon), f"Item {i} should be a rectangle"
        # Check allowed_rotations was reduced for rectangles
        assert item.allowed_rotations <= 2, f"Item {i} rotations should be <=2"

    print("  All items validated as rectangles")
    print("\nCSV loading test PASSED!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RECTANGLE OPTIMIZATION TEST SUITE")
    print("=" * 60)

    results = []

    results.append(("rect_utils", test_rect_utils()))
    results.append(("layout_rect_mode", test_layout_rect_mode()))
    results.append(("initial_solution", test_initial_solution()))
    results.append(("csv_loading", test_csv_loading()))
    results.append(("full_algorithm", test_full_algorithm()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
