# 2D Nesting Solver (Python)

A Python implementation of a **2D Irregular Strip Packing** solver for the cutting stock problem. Optimizes placement of irregular polygons onto rectangular sheets to minimize material waste.

Based on the [2DNesting](https://github.com/lryan599/2DNesting) C++ project.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-GPL--3.0-green.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange.svg)

## Features

- **GOMH Algorithm** - Guided Overlap Minimization Heuristic for efficient packing
- **Rectangle Optimization** - 10-100x faster computation for rectangular shapes
- **Real-time Visualization** - Watch the algorithm progress with live updates
- **Multiple Export Formats** - SVG and TXT output
- **ESICUP Benchmark Support** - Compatible with standard benchmark CSV format

## Quick Start

```bash
# Clone the repository
git clone https://github.com/QuantumRaul/nesting_2d.git
cd nesting_2d

# Install dependencies
pip install -r requirements.txt

# Run the GUI
python main.py
```

## Screenshots

The GUI provides real-time visualization of the nesting process:
- Load CSV files with polygon definitions
- Configure sheet dimensions and runtime limits
- View solution metrics (length, utilization, time)
- Export results to SVG or TXT

## Project Structure

```
nesting_python/
├── main.py                    # Entry point (PyQt6 GUI)
├── requirements.txt           # Dependencies
├── test_rect_optimization.py  # Unit tests
│
├── core/                      # Core algorithm
│   ├── algorithm.py          # GOMH algorithm implementation
│   ├── layout.py             # Layout state & caches
│   ├── nfp.py                # No-Fit Polygon computation
│   ├── geometry.py           # Geometric utilities
│   ├── candidate_points.py   # Placement point generation
│   ├── rect_utils.py         # Rectangle optimizations
│   └── data_types.py         # Item, Sheet, Solution classes
│
├── gui/                       # User Interface
│   ├── main_window.py        # Main application window
│   ├── canvas.py             # Matplotlib visualization
│   └── worker.py             # Background thread
│
├── file_io/                   # Input/Output
│   ├── csv_reader.py         # CSV parser (ESICUP format)
│   └── exporter.py           # SVG/TXT export
│
└── data/csv/                  # Sample data files
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| shapely | >=2.0.0 | 2D geometry operations |
| pyclipper | >=1.3.0 | Minkowski sum (NFP) |
| numpy | >=1.24.0 | Numerical operations |
| pandas | >=2.0.0 | CSV parsing |
| PyQt6 | >=6.5.0 | GUI framework |
| matplotlib | >=3.7.0 | Visualization |

## Input Format (CSV)

The CSV format follows the ESICUP benchmark standard:

```csv
polygon,allowed_rotations,quantity
4 0.0 0.0 10.0 0.0 10.0 5.0 0.0 5.0,4,2
3 0.0 0.0 6.0 0.0 3.0 4.0,2,5
```

**Columns:**
- `polygon`: Vertex count followed by x,y coordinate pairs
- `allowed_rotations`: 1 (none), 2 (0°, 180°), or 4 (0°, 90°, 180°, 270°)
- `quantity`: Number of copies

## Algorithm Overview

The solver uses the **GOMH** (Guided Overlap Minimization Heuristic) approach:

1. **NFP Computation** - No-Fit Polygon via Minkowski sums
2. **IFR Calculation** - Inner-Fit Rectangle for valid placement regions
3. **PD Measurement** - Penetration Depth to quantify overlaps
4. **Iterative Optimization** - Shrink/expand cycles to minimize strip length

### Rectangle Optimization

When all input shapes are rectangles, the solver automatically enables optimized code paths:

- **AABB collision detection** instead of polygon intersection
- **O(1) penetration depth** calculation
- **Reduced rotations** (only 0° and 90° are meaningful)
- **Corner-based candidate points** instead of NFP arrangements

This provides **10-100x performance improvement** for rectangular inputs.

## Usage

### GUI Mode

```bash
python main.py
```

1. Click "Cargar CSV" to load a polygon file
2. Set sheet width and height
3. Configure maximum runtime (seconds)
4. Click "Iniciar Nesting" to start
5. Export results with "Guardar SVG" or "Exportar TXT"

### Programmatic Usage

```python
from core.data_types import Item, Sheet
from core.layout import Layout
from core.algorithm import gomh
from shapely.geometry import Polygon
from threading import Event

# Define items
items = [
    Item(
        polygon=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        quantity=5,
        allowed_rotations=4
    ),
    Item(
        polygon=Polygon([(0, 0), (8, 0), (8, 4), (0, 4)]),
        quantity=3,
        allowed_rotations=4
    )
]

# Create layout
sheet = Sheet(width=100, height=30)
layout = Layout(items, sheet)

# Run algorithm
stop_flag = Event()

def on_progress(solution):
    print(f"Length: {solution.length:.2f}, Utilization: {solution.utilization:.2%}")

gomh(layout, max_time=60, progress_callback=on_progress, stop_flag=stop_flag)
```

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rinc` | 0.01 | Strip length increase ratio |
| `rdec` | 0.04 | Strip length decrease ratio |
| `max_iterations` | 50 | Max iterations per optimization round |
| `max_pd_cache` | 200000 | Penetration depth cache size |

## Testing

Run the test suite:

```bash
python test_rect_optimization.py
```

Tests include:
- Rectangle detection and dimension extraction
- NFP and penetration depth calculations
- Collision detection
- Candidate point generation
- Full algorithm integration

## Export Formats

### SVG
Visual representation with:
- Colored polygons (16-color palette)
- Sheet boundaries
- Solution metadata

### TXT
Detailed coordinates:
```
Solution Length: 85.50
Utilization: 78.45%
Part 0: x=0.00, y=0.00, rotation=0
Part 1: x=10.50, y=0.00, rotation=1
...
```

## Performance Tips

1. **Use rectangular shapes** when possible for 10-100x speedup
2. **Increase max runtime** for better solutions on complex problems
3. **Reduce rotations** if orientation doesn't matter (set `allowed_rotations=1`)
4. **Pre-sort items** by area (largest first) for better initial solutions

## License

GPL-3.0 - See [LICENSE](LICENSE) for details.

This license is required due to the CGAL dependency in the original C++ implementation.

## Acknowledgments

- Original C++ implementation: [2DNesting](https://github.com/lryan599/2DNesting)
- ESICUP benchmark problems: [ESICUP](https://www.euro-online.org/websites/esicup/)

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request
