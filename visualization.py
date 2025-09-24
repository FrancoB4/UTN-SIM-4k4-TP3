import pandas as pd
from PyQt5.QtCore import Qt, QAbstractTableModel


class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole): # type: ignore
        if not index.isValid():
            return None
        
        value = self._df.iloc[index.row(), index.column()]
        
        if role == Qt.DisplayRole: # type: ignore
            if isinstance(value, (int, float)) and not pd.isna(value):
                if isinstance(value, float) and not value.is_integer():
                    return f"{value:.{4}f}"
                return str(int(value))
            return str(value)
        elif role == Qt.UserRole: # type: ignore
            return value
        
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole): # type: ignore
        if role == Qt.DisplayRole: # type: ignore
            if orientation == Qt.Horizontal: # type: ignore
                return str(self._df.columns[section])
            if orientation == Qt.Vertical: # type: ignore
                return str(self._df.index[section])
        return None
