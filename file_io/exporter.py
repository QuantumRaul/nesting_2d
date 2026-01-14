"""
Export functions for nesting results.

Supports:
- SVG export with visualization
- Text export with polygon coordinates
"""
from typing import List
from shapely.geometry import Polygon

from core.data_types import Solution, TransformedShape


def export_svg(solution: Solution, sheet_width: float, sheet_height: float,
               filepath: str) -> None:
    """
    Export the solution to an SVG file.

    Args:
        solution: The solution to export
        sheet_width: Width of the sheet
        sheet_height: Height of the sheet
        filepath: Output file path
    """
    # SVG viewbox with some margin
    margin = 10
    viewbox_width = solution.length + 2 * margin
    viewbox_height = sheet_height + 2 * margin

    # Colors for different items
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1'
    ]

    svg_parts = []

    # SVG header
    svg_parts.append(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="{-margin} {-margin} {viewbox_width} {viewbox_height}"
     width="{viewbox_width * 2}" height="{viewbox_height * 2}">
''')

    # Background (sheet)
    svg_parts.append(f'''  <rect x="0" y="0" width="{solution.length}" height="{sheet_height}"
        fill="#f0f0f0" stroke="#333" stroke-width="0.5"/>
''')

    # Draw each polygon
    for i, shape in enumerate(solution.shapes):
        polygon = shape.transformed
        color = colors[shape.item_idx % len(colors)]

        # Get polygon coordinates
        coords = list(polygon.exterior.coords)
        points_str = " ".join([f"{x},{y}" for x, y in coords])

        svg_parts.append(f'''  <polygon points="{points_str}"
          fill="{color}" fill-opacity="0.7"
          stroke="#333" stroke-width="0.3"/>
''')

    # Add info text
    svg_parts.append(f'''  <text x="5" y="{sheet_height + margin - 2}"
        font-size="3" fill="#333">
    Length: {solution.length:.2f} | Utilization: {solution.utilization:.1%} | Time: {solution.time:.1f}s
  </text>
''')

    # SVG footer
    svg_parts.append('</svg>\n')

    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(''.join(svg_parts))


def export_txt(solution: Solution, sheet_width: float, sheet_height: float,
               filepath: str) -> None:
    """
    Export the solution to a text file.

    Args:
        solution: The solution to export
        sheet_width: Width of the sheet
        sheet_height: Height of the sheet
        filepath: Output file path
    """
    lines = []

    lines.append("2D Nesting Solution")
    lines.append("=" * 50)
    lines.append(f"Sheet size: {sheet_width:.2f} x {sheet_height:.2f}")
    lines.append(f"Used length: {solution.length:.2f}")
    lines.append(f"Utilization: {solution.utilization:.2%}")
    lines.append(f"Time: {solution.time:.2f} seconds")
    lines.append(f"Number of parts: {len(solution.shapes)}")
    lines.append("")
    lines.append("Placed parts:")
    lines.append("-" * 50)

    for i, shape in enumerate(solution.shapes):
        polygon = shape.transformed
        coords = list(polygon.exterior.coords)[:-1]  # Exclude closing point

        lines.append(f"\nPart {i + 1}:")
        lines.append(f"  Item index: {shape.item_idx}")
        lines.append(f"  Rotation: {shape.rotation * 90}°")
        lines.append(f"  Position: ({shape.translate_x:.4f}, {shape.translate_y:.4f})")
        lines.append(f"  Vertices: {len(coords)}")

        for j, (x, y) in enumerate(coords):
            lines.append(f"    V{j + 1}: ({x:.4f}, {y:.4f})")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def polygon_to_svg_path(polygon: Polygon) -> str:
    """
    Convert a Shapely polygon to SVG path data.

    Args:
        polygon: The polygon

    Returns:
        SVG path data string
    """
    coords = list(polygon.exterior.coords)
    if not coords:
        return ""

    path_data = f"M {coords[0][0]},{coords[0][1]}"
    for x, y in coords[1:]:
        path_data += f" L {x},{y}"
    path_data += " Z"

    return path_data
