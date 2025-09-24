import matplotlib
import pandas as pd

from simulator import ConcurrenceProbabilities, Probability, Simulator
from visualization import PandasModel
matplotlib.use('Qt5Agg')


from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QLineEdit, QPushButton, QApplication, QLabel
)

from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt
from generators import *


class CopyableTableView(QTableView):
    def keyPressEvent(self, e):
        if e.matches(QKeySequence.Copy):  # type: ignore
            self.copySelectionToClipboard()
        else:
            super().keyPressEvent(e)

    def copySelectionToClipboard(self):
        selection = self.selectionModel()
        if not selection.hasSelection():  # type: ignore
            return

        indexes = selection.selectedIndexes()  # type: ignore
        if not indexes:
            return

        indexes = sorted(indexes, key=lambda x: (x.row(), x.column()))
        
        rows = []
        current_row = indexes[0].row()
        row_data = []

        # Get column headers for selected columns
        selected_columns = set(idx.column() for idx in indexes)
        headers = []
        for col in sorted(selected_columns):
            if self.model():
                header = self.model().headerData(col, 1)  # type: ignore
                headers.append(str(header) if header else f"Column {col}")

        # Add headers as first row
        if headers and indexes[0].row() == 0:
            rows.append('\t'.join(headers))
        
        for idx in indexes:
            if idx.row() != current_row:
                rows.append('\t'.join(row_data))
                row_data = []
                current_row = idx.row()
            original_value = self.model().data(idx, Qt.UserRole) # type: ignore
            if original_value is not None:
                row_data.append(str(original_value))
            else:
                # Fallback al valor formateado si no está disponible el original
                row_data.append(str(idx.data()))
        rows.append('\t'.join(row_data))

        clipboard_text = '\n'.join(rows)

        QApplication.clipboard().setText(clipboard_text.replace('.', ','))  # type: ignore


class RightPanel(QWidget):
    def __init__(self, x, label: str = '', parent=None):
        super().__init__(parent)
        
        self.x = x
        self.label = label
        
        self.setWindowTitle('Visualización de la simulación')
        self.setGeometry(100, 100, 400, 600)

        layout = QVBoxLayout(self)
        
        self.title = QLabel('Resultados de la simulación', self)
        layout.addWidget(self.title)
        
        self.table = CopyableTableView()
        layout.addWidget(self.table)

        self.setLayout(layout)
    
    def on_update_table(self, model: PandasModel):
        self.table.setModel(model)
        self.table.resizeColumnsToContents()


class ProbabilityDistributionEditor(QWidget):
    def __init__(self, n_rows: int = 10, labels: list[str] = [], parent=None):
        super().__init__(parent)
        
        if not len(labels) == n_rows:
            raise ValueError("La longitud de las etiquetas debe ser igual al número de filas.")
        
        self.n_rows = n_rows
        self.probability_inputs = []
        
        layout = QVBoxLayout(self)
        
        for i in range(n_rows):
            row_layout = QHBoxLayout()

            label = QLabel(labels[i], self)
            row_layout.addWidget(label)
            
            probability_input = QLineEdit(self)
            probability_input.setPlaceholderText('0.0 - 1.0')
            row_layout.addWidget(probability_input)
            
            self.probability_inputs.append(probability_input)
            layout.addLayout(row_layout)
        
        self.probability_table_title = QLabel('Distribución de Probabilidades', self)
        layout.addWidget(self.probability_table_title)

        self.probability_table = CopyableTableView()
        layout.addWidget(self.probability_table)
        
        self.probability_table.setModel(PandasModel(pd.DataFrame()))
        
        self.setLayout(layout)

    def check_inputs(self) -> tuple[bool, str]:
        total = 0.0
        for input_field in self.probability_inputs:
            try:
                value = float(input_field.text()) if input_field.text() else 0.0
                if value < 0.0 or value > 1.0:
                    return False, f"Probabilidad inválida: {value}. Debe estar entre 0 y 1."
                total += value
            except ValueError:
                return False, f"Entrada inválida: {input_field.text()}. Debe ser un número."
        
        try:
            probabilities = ConcurrenceProbabilities(
                self.n_rows, 
                [
                    Probability(38 + i, float(self.probability_inputs[i].text()) 
                                if self.probability_inputs[i].text() else 0.0) 
                    for i in range(self.n_rows)
                ])
        except Exception as e:
            return False, "La suma de las probabilidades debe ser 1."
        
        return True, ""
    
    def get_probabilities(self):
        if self.check_inputs()[0]:
            probabilities = []
            for input_field in self.probability_inputs:
                try:
                    value = float(input_field.text()) if input_field.text() else 0.0
                    probabilities.append(Probability(38 + len(probabilities), value))
                except ValueError:
                    probabilities.append(Probability(38 + len(probabilities), 0.0))

            res = ConcurrenceProbabilities(self.n_rows, probabilities)

            self.probability_table.setModel(res.to_model())

            return res
        else:
            raise ValueError("Las entradas no son válidas.")

    def set_probabilities(self, probabilities):
        for i, probability in enumerate(probabilities):
            if i < len(self.probability_inputs):
                self.probability_inputs[i].setText(str(probability))


class LeftPanel(QWidget):
    def __init__(self, update_simulation, n_rows: int = 10, parent=None):
        super().__init__(parent)

        self.update_simulation = update_simulation

        self.setWindowTitle('Configuración de la simulación')
        self.setGeometry(100, 100, 400, 600)

        layout = QVBoxLayout(self)
        
        self._add_configuration(layout)

        self.probability_distribution_editor = ProbabilityDistributionEditor(
            n_rows=n_rows,
            labels=[f'Número de pasajeros {38 + i}' for i in range(n_rows)]
        )
        layout.addWidget(self.probability_distribution_editor)

        self.error_label = QLabel('', self)
        self.error_label.setStyleSheet('color: red;')
        layout.addWidget(self.error_label)
        
        self.generate_button = QPushButton('Generar variable aleatoria', self)
        self.generate_button.clicked.connect(self.on_generate)
        layout.addWidget(self.generate_button)

        self.setLayout(layout)
    
    def _add_configuration(self, layout: QVBoxLayout):
        self.n_input = QLineEdit(self)
        self.n_input.setPlaceholderText('Tamaño de la muestra (n)')
        layout.addWidget(self.n_input)
        
        self.passage_price = QLineEdit(self)
        self.passage_price.setPlaceholderText('Ingreso por pasaje vendido')
        layout.addWidget(self.passage_price)
        
        # self.plain_capacity = QLineEdit(self)
        # self.plain_capacity.setPlaceholderText('Cantidad de asientos en el avión')
        # layout.addWidget(self.plain_capacity)
        
        # self.oversell_limit = QLineEdit(self)
        # self.oversell_limit.setPlaceholderText('Cantidad máxima de pasajes a vender')
        # layout.addWidget(self.oversell_limit)
        
        self.oversell_cost = QLineEdit(self)
        self.oversell_cost.setPlaceholderText('Costo por pasajero que no puede viajar')
        layout.addWidget(self.oversell_cost)
    
    def _check_inputs(self):
        if not self.n_input.text().replace('-', '').isdigit():
            self.error_label.setText('Error: El tamaño de la muestra debe ser un número.')
            return False
        
        if not int(self.n_input.text()) > 0:
            self.error_label.setText('Error: El tamaño de la muestra debe ser mayor que 0.')
            return False
        
        if not int(self.n_input.text()) <= 100_000:
            self.error_label.setText('Error: El tamaño de la muestra debe ser menor que 100.000.')
            return False
        
        if not self.passage_price.text().replace('.', '', 1).isdigit():
            self.error_label.setText('Error: El ingreso por pasaje vendido debe ser un número.')
            return False
        
        if not float(self.passage_price.text()) > 0:
            self.error_label.setText('Error: El ingreso por pasaje vendido debe ser mayor que 0.')
            return False

        # if not self.plain_capacity.text().replace('.', '', 1).isdigit():
        #     self.error_label.setText('Error: La capacidad del avión debe ser un número.')
        #     return False

        # if not int(self.plain_capacity.text()) > 0:
        #     self.error_label.setText('Error: La capacidad del avión debe ser mayor que 0.')
        #     return False

        # if not self.oversell_limit.text().replace('.', '', 1).isdigit():
        #     self.error_label.setText('Error: El límite de sobreventa debe ser un número.')
        #     return False

        # if not int(self.oversell_limit.text()) > 0:
        #     self.error_label.setText('Error: El límite de sobreventa debe ser mayor que 0.')
        #     return False

        # if not int(self.oversell_limit.text()) >= int(self.plain_capacity.text()):
        #     self.error_label.setText('Error: El límite de sobreventa debe ser mayor o igual a la capacidad del avión.')
        #     return False
        
        if self.oversell_cost.text() and not self.oversell_cost.text().replace('.', '', 1).isdigit():
            self.error_label.setText('Error: El costo por pasajero que no puede viajar debe ser un número.')
            return False
        
        if self.oversell_cost.text() and not float(self.oversell_cost.text()) >= 0:
            self.error_label.setText('Error: El costo por pasajero que no puede viajar debe ser mayor o igual a 0.')
            return False
        
        if not self.n_input.text() or not self.passage_price.text():
            self.error_label.setText('Error: Todos los campos son obligatorios.')
            return False
        
        if not (self.probability_distribution_editor.check_inputs()[0]):
            self.error_label.setText('Error: ' + self.probability_distribution_editor.check_inputs()[1])  # type: ignore
            return False
        
        self.error_label.setText('')
        
        return True
    
    def _get_data(self):
        raise NotImplementedError('This method should be implemented in subclasses.')

    def on_generate(self):
        if not self._check_inputs():
            return
        
        try:
            probabilities = self.probability_distribution_editor.get_probabilities()
        except ValueError as e:
            self.error_label.setText('Error: ' + str(e))  # type: ignore
            return
        
        self.update_simulation(
            int(self.n_input.text()),
            probabilities,
            float(self.passage_price.text()),
            float(self.oversell_cost.text()) if self.oversell_cost.text() else 0.0
        )


class Tab(QWidget):
    def __init__(self, n_rows: int, parent=None):
        super().__init__(parent)

        self.simulator = Simulator(42, ConcurrenceProbabilities.default_probabilities())
        self.left_panel = LeftPanel(self.update_simulation, n_rows=n_rows)

        layout = QHBoxLayout(self)
        layout.addWidget(self.left_panel, 34)  # LeftPanel ocupa 34% de la ventana
        
        self.right_panel = RightPanel([])
        
        layout.addWidget(self.right_panel, 66)  # RightPanel ocupa 66% de la ventana

        self.setLayout(layout)

    def update_simulation(self, n: int, probabilities, passage_price: float, oversell_cost: float = 0.0):
        self.simulator.update_params(n=n, probabilities=probabilities, passage_price=passage_price, oversell_cost=oversell_cost)
        self.right_panel.on_update_table(self.simulator.to_model())
    
    def update_dist_table(self, counts, bin_edges):
        self.left_panel.update_dist_table(counts, bin_edges)
