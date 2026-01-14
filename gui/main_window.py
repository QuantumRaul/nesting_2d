"""
Main window for the 2D Nesting application.
"""
import os
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QFileDialog, QMessageBox, QStatusBar,
    QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .canvas import NestingCanvas
from .worker import NestingWorker
from core.data_types import Item, Sheet, Solution
from file_io.csv_reader import read_csv, get_total_area
from file_io.exporter import export_svg, export_txt


class MainWindow(QMainWindow):
    """
    Main application window for 2D Nesting.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("2D Nesting - Python")
        self.setMinimumSize(1200, 800)

        # State
        self.items: list[Item] = []
        self.current_solution: Optional[Solution] = None
        self.worker: Optional[NestingWorker] = None

        # Setup UI
        self.setup_ui()
        self.setup_statusbar()

    def setup_ui(self) -> None:
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel (controls)
        left_panel = self.create_control_panel()

        # Right panel (canvas)
        self.canvas = NestingCanvas(self, width=10, height=8)

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

    def create_control_panel(self) -> QWidget:
        """Create the control panel with input fields and buttons."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # File section
        file_group = QGroupBox("Archivo de entrada")
        file_layout = QVBoxLayout()

        self.file_label = QLabel("No se ha cargado ningún archivo")
        self.file_label.setWordWrap(True)

        load_btn = QPushButton("Cargar CSV...")
        load_btn.clicked.connect(self.load_csv)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(load_btn)
        file_group.setLayout(file_layout)

        # Sheet configuration
        sheet_group = QGroupBox("Configuración de la hoja")
        sheet_layout = QVBoxLayout()

        # Width
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Ancho:"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1, 100000)
        self.width_spin.setValue(100)
        self.width_spin.setDecimals(2)
        width_layout.addWidget(self.width_spin)
        sheet_layout.addLayout(width_layout)

        # Height
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Alto:"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 100000)
        self.height_spin.setValue(40)
        self.height_spin.setDecimals(2)
        height_layout.addWidget(self.height_spin)
        sheet_layout.addLayout(height_layout)

        sheet_group.setLayout(sheet_layout)

        # Algorithm parameters
        algo_group = QGroupBox("Parámetros del algoritmo")
        algo_layout = QVBoxLayout()

        # Max time
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Tiempo máx (s):"))
        self.time_spin = QSpinBox()
        self.time_spin.setRange(10, 3600)
        self.time_spin.setValue(60)
        time_layout.addWidget(self.time_spin)
        algo_layout.addLayout(time_layout)

        algo_group.setLayout(algo_layout)

        # Control buttons
        btn_group = QGroupBox("Control")
        btn_layout = QVBoxLayout()

        self.start_btn = QPushButton("Iniciar Nesting")
        self.start_btn.clicked.connect(self.start_nesting)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        self.stop_btn = QPushButton("Detener")
        self.stop_btn.clicked.connect(self.stop_nesting)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_group.setLayout(btn_layout)

        # Results section
        results_group = QGroupBox("Resultados")
        results_layout = QVBoxLayout()

        self.length_label = QLabel("Longitud: -")
        self.util_label = QLabel("Utilización: -")
        self.time_label = QLabel("Tiempo: -")
        self.parts_label = QLabel("Piezas: -")

        font = QFont()
        font.setPointSize(10)
        for label in [self.length_label, self.util_label, self.time_label, self.parts_label]:
            label.setFont(font)
            results_layout.addWidget(label)

        results_group.setLayout(results_layout)

        # Export buttons
        export_group = QGroupBox("Exportar")
        export_layout = QVBoxLayout()

        svg_btn = QPushButton("Exportar SVG...")
        svg_btn.clicked.connect(self.export_svg)

        txt_btn = QPushButton("Exportar TXT...")
        txt_btn.clicked.connect(self.export_txt)

        export_layout.addWidget(svg_btn)
        export_layout.addWidget(txt_btn)
        export_group.setLayout(export_layout)

        # Add all groups to layout
        layout.addWidget(file_group)
        layout.addWidget(sheet_group)
        layout.addWidget(algo_group)
        layout.addWidget(btn_group)
        layout.addWidget(results_group)
        layout.addWidget(export_group)
        layout.addStretch()

        return panel

    def setup_statusbar(self) -> None:
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)

        self.statusbar.showMessage("Listo. Carga un archivo CSV para comenzar.")

    def load_csv(self) -> None:
        """Load items from a CSV file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir archivo CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not filepath:
            return

        try:
            self.items = read_csv(filepath)
            total_area = get_total_area(self.items)
            n_parts = sum(item.quantity for item in self.items)

            # Update UI
            filename = os.path.basename(filepath)
            self.file_label.setText(
                f"Archivo: {filename}\n"
                f"Items: {len(self.items)}\n"
                f"Piezas totales: {n_parts}\n"
                f"Área total: {total_area:.2f}"
            )

            # Update parts label
            self.parts_label.setText(f"Piezas: {n_parts}")

            # Enable start button
            self.start_btn.setEnabled(True)

            # Show preview
            self.canvas.draw_items(self.items)

            self.statusbar.showMessage(f"Cargado: {filename} con {n_parts} piezas")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar archivo:\n{e}")

    def start_nesting(self) -> None:
        """Start the nesting algorithm."""
        if not self.items:
            QMessageBox.warning(self, "Aviso", "Primero carga un archivo CSV")
            return

        # Get parameters
        sheet_width = self.width_spin.value()
        sheet_height = self.height_spin.value()
        max_time = self.time_spin.value()

        # Create sheet
        sheet = Sheet(width=sheet_width, height=sheet_height)

        # Update canvas
        self.canvas.set_sheet_size(sheet_width, sheet_height)

        # Create and start worker
        self.worker = NestingWorker(self.items, sheet, max_time)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(True)
        self.statusbar.showMessage("Ejecutando algoritmo...")

        # Start
        self.worker.start()

    def stop_nesting(self) -> None:
        """Stop the nesting algorithm."""
        if self.worker is not None:
            self.worker.stop()
            self.statusbar.showMessage("Deteniendo algoritmo...")

    def on_progress(self, solution: Solution) -> None:
        """Handle progress update from worker."""
        self.current_solution = solution

        # Update canvas
        self.canvas.draw_solution(solution)

        # Update labels
        self.length_label.setText(f"Longitud: {solution.length:.2f}")
        self.util_label.setText(f"Utilización: {solution.utilization:.1%}")
        self.time_label.setText(f"Tiempo: {solution.time:.1f}s")

        self.statusbar.showMessage(
            f"Nueva solución: {solution.length:.2f} ({solution.utilization:.1%})"
        )

    def on_finished(self, solution: Solution) -> None:
        """Handle algorithm completion."""
        self.current_solution = solution

        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        # Final update
        self.canvas.draw_solution(solution)
        self.length_label.setText(f"Longitud: {solution.length:.2f}")
        self.util_label.setText(f"Utilización: {solution.utilization:.1%}")
        self.time_label.setText(f"Tiempo: {solution.time:.1f}s")

        self.statusbar.showMessage(
            f"Completado: Longitud = {solution.length:.2f}, "
            f"Utilización = {solution.utilization:.1%}"
        )

        self.worker = None

    def on_error(self, error_msg: str) -> None:
        """Handle error from worker."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        QMessageBox.critical(self, "Error", f"Error en el algoritmo:\n{error_msg}")
        self.statusbar.showMessage("Error en el algoritmo")

        self.worker = None

    def export_svg(self) -> None:
        """Export current solution to SVG."""
        if self.current_solution is None:
            QMessageBox.warning(self, "Aviso", "No hay solución para exportar")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar como SVG",
            "nesting_result.svg",
            "SVG Files (*.svg);;All Files (*)"
        )

        if filepath:
            try:
                export_svg(
                    self.current_solution,
                    self.width_spin.value(),
                    self.height_spin.value(),
                    filepath
                )
                self.statusbar.showMessage(f"Exportado: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar:\n{e}")

    def export_txt(self) -> None:
        """Export current solution to TXT."""
        if self.current_solution is None:
            QMessageBox.warning(self, "Aviso", "No hay solución para exportar")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar como TXT",
            "nesting_result.txt",
            "Text Files (*.txt);;All Files (*)"
        )

        if filepath:
            try:
                export_txt(
                    self.current_solution,
                    self.width_spin.value(),
                    self.height_spin.value(),
                    filepath
                )
                self.statusbar.showMessage(f"Exportado: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar:\n{e}")

    def closeEvent(self, event) -> None:
        """Handle window close."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)  # Wait up to 2 seconds
        event.accept()
