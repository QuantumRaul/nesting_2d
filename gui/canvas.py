"""
Matplotlib canvas for polygon visualization in PyQt.
"""
from typing import List, Optional
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

from core.data_types import Solution, TransformedShape, Item


class NestingCanvas(FigureCanvas):
    """
    Canvas widget for displaying nesting results.
    """

    # Colors for different items
    COLORS = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1',
        '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'
    ]

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """
        Initialize the canvas.

        Args:
            parent: Parent widget
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)

        # Configuration
        self.sheet_width = 100
        self.sheet_height = 50

        # Initial setup
        self.setup_axes()

    def setup_axes(self) -> None:
        """Configure the axes for display."""
        self.axes.set_aspect('equal')
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.grid(True, alpha=0.3)

    def clear(self) -> None:
        """Clear the canvas."""
        self.axes.clear()
        self.setup_axes()
        self.draw()

    def set_sheet_size(self, width: float, height: float) -> None:
        """
        Set the sheet dimensions.

        Args:
            width: Sheet width
            height: Sheet height
        """
        self.sheet_width = width
        self.sheet_height = height

    def draw_sheet(self, length: Optional[float] = None) -> None:
        """
        Draw the sheet rectangle.

        Args:
            length: Optional used length (defaults to full width)
        """
        if length is None:
            length = self.sheet_width

        # Draw sheet outline
        sheet_rect = MplPolygon(
            [(0, 0), (length, 0), (length, self.sheet_height), (0, self.sheet_height)],
            fill=True,
            facecolor='#f5f5f5',
            edgecolor='#333333',
            linewidth=1.5
        )
        self.axes.add_patch(sheet_rect)

        # Set axis limits with padding
        padding = max(length, self.sheet_height) * 0.05
        self.axes.set_xlim(-padding, length + padding)
        self.axes.set_ylim(-padding, self.sheet_height + padding)

    def draw_items(self, items: List[Item]) -> None:
        """
        Draw input items (preview before nesting).

        Args:
            items: List of items to display
        """
        self.clear()

        # Calculate grid layout
        n_items = sum(item.quantity for item in items)
        cols = int(np.ceil(np.sqrt(n_items)))

        # Find max item size
        max_size = 0
        for item in items:
            minx, miny, maxx, maxy = item.polygon.bounds
            max_size = max(max_size, maxx - minx, maxy - miny)

        cell_size = max_size * 1.2

        # Draw each item
        idx = 0
        for item_idx, item in enumerate(items):
            color = self.COLORS[item_idx % len(self.COLORS)]

            for _ in range(item.quantity):
                row = idx // cols
                col = idx % cols

                # Offset for this cell
                offset_x = col * cell_size
                offset_y = row * cell_size

                # Get polygon coords and offset
                coords = list(item.polygon.exterior.coords)
                minx, miny, _, _ = item.polygon.bounds
                offset_coords = [(x - minx + offset_x, y - miny + offset_y)
                                 for x, y in coords]

                patch = MplPolygon(
                    offset_coords,
                    fill=True,
                    facecolor=color,
                    edgecolor='#333333',
                    linewidth=0.5,
                    alpha=0.7
                )
                self.axes.add_patch(patch)

                idx += 1

        # Set axis limits
        total_width = cols * cell_size
        total_height = ((n_items - 1) // cols + 1) * cell_size
        padding = cell_size * 0.1
        self.axes.set_xlim(-padding, total_width + padding)
        self.axes.set_ylim(-padding, total_height + padding)
        self.axes.set_title(f'Input: {n_items} parts from {len(items)} items')

        self.draw()

    def draw_solution(self, solution: Solution) -> None:
        """
        Draw a nesting solution.

        Args:
            solution: The solution to display
        """
        self.clear()

        # Draw sheet
        self.draw_sheet(solution.length)

        # Draw each placed polygon
        for shape in solution.shapes:
            polygon = shape.transformed
            coords = list(polygon.exterior.coords)
            color = self.COLORS[shape.item_idx % len(self.COLORS)]

            patch = MplPolygon(
                coords,
                fill=True,
                facecolor=color,
                edgecolor='#333333',
                linewidth=0.5,
                alpha=0.7
            )
            self.axes.add_patch(patch)

        # Update title
        self.axes.set_title(
            f'Length: {solution.length:.2f} | '
            f'Utilization: {solution.utilization:.1%} | '
            f'Time: {solution.time:.1f}s'
        )

        self.draw()

    def draw_shapes(self, shapes: List[TransformedShape],
                    sheet_length: float) -> None:
        """
        Draw a list of transformed shapes.

        Args:
            shapes: List of shapes to display
            sheet_length: Current sheet length
        """
        self.clear()

        # Draw sheet
        self.draw_sheet(sheet_length)

        # Draw each polygon
        for shape in shapes:
            polygon = shape.transformed
            coords = list(polygon.exterior.coords)
            color = self.COLORS[shape.item_idx % len(self.COLORS)]

            patch = MplPolygon(
                coords,
                fill=True,
                facecolor=color,
                edgecolor='#333333',
                linewidth=0.5,
                alpha=0.7
            )
            self.axes.add_patch(patch)

        self.draw()

    def update_solution(self, solution: Solution) -> None:
        """
        Update the display with a new solution (thread-safe).

        Args:
            solution: The new solution to display
        """
        self.draw_solution(solution)
