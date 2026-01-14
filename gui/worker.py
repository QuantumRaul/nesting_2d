"""
Worker thread for running the nesting algorithm without blocking the UI.
"""
from typing import List
from threading import Event

from PyQt6.QtCore import QThread, pyqtSignal

from core.data_types import Item, Sheet, Solution
from core.layout import Layout
from core.algorithm import gomh


class NestingWorker(QThread):
    """
    Worker thread that runs the GOMH nesting algorithm.

    Emits signals for progress updates and completion.
    """

    # Signals
    progress = pyqtSignal(Solution)  # Emitted when a better solution is found
    finished_signal = pyqtSignal(Solution)  # Emitted when algorithm completes
    error = pyqtSignal(str)  # Emitted on error

    def __init__(self, items: List[Item], sheet: Sheet, max_time: int = 60):
        """
        Initialize the worker.

        Args:
            items: List of items to nest
            sheet: The sheet to place items on
            max_time: Maximum time in seconds
        """
        super().__init__()

        self.items = items
        self.sheet = sheet
        self.max_time = max_time

        # Stop flag for clean termination
        self.stop_flag = Event()

        # Best solution found
        self.best_solution: Solution = None

    def run(self) -> None:
        """
        Run the nesting algorithm.

        Called when thread.start() is invoked.
        """
        try:
            # Create layout
            layout = Layout(self.items, self.sheet)

            # Define progress callback
            def on_progress(solution: Solution):
                self.best_solution = solution
                self.progress.emit(solution)

            # Run algorithm
            gomh(layout, self.max_time, on_progress, self.stop_flag)

            # Emit final result
            if self.best_solution is not None:
                self.finished_signal.emit(self.best_solution)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self) -> None:
        """
        Request the algorithm to stop.

        The algorithm will stop at the next iteration check.
        """
        self.stop_flag.set()
