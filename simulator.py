import numpy as np
import pandas as pd
from visualization import PandasModel


class Probability:
    def __init__(self, value: int, probability: float):
        self.value = value
        self.probability = probability
        self.low_limit = 0.0
        self.up_limit = 0.0


class ConcurrenceProbabilities:
    def __init__(self, n: int, probabilities: list[Probability]):
        self.n = n
        self.probabilities = self._generate_concurrence_probabilities(probabilities)

    def _check_probabilities_consistency(self, probabilities: list[Probability]) -> bool:
        total_prob = sum(p.probability for p in probabilities)
        return np.isclose(total_prob, 1.0) # type: ignore
    
    def _generate_concurrence_probabilities(self, probabilities: list[Probability]) -> list[Probability]:
        current_sum = 0.0
        for i, prob in enumerate(probabilities):
            if i == 0:
                prob.low_limit = 0.0
                prob.up_limit = prob.probability - 1e-4
            else:
                prob.low_limit = current_sum
                prob.up_limit = prob.low_limit + prob.probability - 1e-4

            current_sum += prob.probability
            
        if not np.isclose(current_sum, 1.0):
            raise ValueError("The sum of probabilities must be 1.")
        return probabilities
    
    def update_probabilities(self, probabilities: list[Probability]):
        if not self._check_probabilities_consistency(probabilities):
            raise ValueError("The sum of probabilities must be 1.")
        self.probabilities = self._generate_concurrence_probabilities(probabilities)
        
    def to_dataframe(self) -> pd.DataFrame:
        data = {
            'Value': [p.value for p in self.probabilities],
            'Probability': [p.probability for p in self.probabilities],
            'Low Limit': [p.low_limit for p in self.probabilities],
            'Up Limit': [p.up_limit for p in self.probabilities],
        }
        return pd.DataFrame(data)
    
    def to_model(self) -> PandasModel:
        return PandasModel(self.to_dataframe())
    
    @classmethod
    def default_probabilities(cls) -> 'ConcurrenceProbabilities':
        probabilities = [
            Probability(38, 0.05),
            Probability(39, 0.25),
            Probability(40, 0.45),
            Probability(41, 0.15),
            Probability(42, 0.05),
            Probability(43, 0.05)
        ]
        return ConcurrenceProbabilities(n=43, probabilities=probabilities)
    
    @classmethod
    def default_probabilities_42(cls) -> 'ConcurrenceProbabilities':
        probabilities = [
            Probability(38, 0.10),
            Probability(39, 0.25),
            Probability(40, 0.40),
            Probability(41, 0.15),
            Probability(42, 0.10),
        ]
        return ConcurrenceProbabilities(n=42, probabilities=probabilities)
    
    @classmethod
    def default_probabilities_44(cls) -> 'ConcurrenceProbabilities':
        probabilities = [
            Probability(38, 0.0),
            Probability(39, 0.05),
            Probability(40, 0.2),
            Probability(41, 0.45),
            Probability(42, 0.15),
            Probability(43, 0.05),
            Probability(44, 0.10),
        ]
        return ConcurrenceProbabilities(n=44, probabilities=probabilities)


class Simulator:
    def __init__(self, n: int, probabilities: ConcurrenceProbabilities, passage_price: float = 100, plain_capacity: int = 40, oversell_limit: int = 43, oversell_cost: float = 150):
        self.n = n
        self.probabilities = probabilities
        self.passage_price = passage_price
        self.plain_capacity = plain_capacity
        self.oversell_limit = oversell_limit
        self.oversell_cost = oversell_cost
        self.total_revenue = 0
        self.total_cost = 0
        self.total_profit = 0
        self.simulations = pd.DataFrame(columns=[
            'Vuelo', 'Número Aleatorio', 'Pasajeros presentes', 'Ingresos', 'Costo sobreventa', 'Utilidad',
            'Ingresos Acumulados', 'Costo Acumulado', 'Utilidad Acumulada',
            'Ingresos Promedio', 'Costo Promedio', 'Utilidad Promedio'
            ])
    
    def _get_passengers_presented(self, rnd: float, probabilities: ConcurrenceProbabilities) -> int:
        for i, prob in enumerate(probabilities.probabilities):
            if prob.low_limit <= rnd < prob.up_limit + 1e-4:
                return prob.value
        
        raise ValueError("Random number out of range")
    
    def _get_row_data(self, flight):
        rnd_number = np.random.rand()
        passengers = self._get_passengers_presented(rnd_number, self.probabilities)
        revenue = passengers * self.passage_price if passengers <= self.plain_capacity else self.plain_capacity * self.passage_price
        cost = (passengers - self.plain_capacity) * self.oversell_cost if passengers > self.plain_capacity else 0
        profit = revenue - cost
        
        self.total_revenue += revenue
        self.total_cost += cost
        self.total_profit += profit
        
        avg_revenue = self.total_revenue / flight
        avg_cost = self.total_cost / flight
        avg_profit = self.total_profit / flight

        return flight, round(rnd_number, 4), passengers, revenue, cost, profit, self.total_revenue, \
            self.total_cost, self.total_profit, round(avg_revenue, 4), round(avg_cost, 4), round(avg_profit, 4)

    def run_simulation(self):
        data = np.array([self._get_row_data(flight) for flight in range(1, self.n + 1)])
        self.simulations = pd.DataFrame(data, columns=[
            'Vuelo', 'Número Aleatorio', 'Pasajeros presentes', 'Ingresos', 'Costo sobreventa', 'Utilidad',
            'Ingresos Acumulados', 'Costo Acumulado', 'Utilidad Acumulada',
            'Ingresos Promedio', 'Costo Promedio', 'Utilidad Promedio'
            ])

    def update_params(self, n: int | None = None, probabilities: ConcurrenceProbabilities | None = None, passage_price: float | None = None, plain_capacity: int | None = None, oversell_limit: int | None = None, oversell_cost: float | None = None):
        if n is not None:
            self.n = n
        if probabilities is not None:
            self.probabilities = probabilities
        if passage_price is not None:
            self.passage_price = passage_price
        if plain_capacity is not None:
            self.plain_capacity = plain_capacity
        if oversell_limit is not None:
            self.oversell_limit = oversell_limit
        if oversell_cost is not None:
            self.oversell_cost = oversell_cost
        
        
        self.total_revenue = 0
        self.total_cost = 0
        self.total_profit = 0
        self.simulations = pd.DataFrame(columns=[
            'Vuelo', 'Número Aleatorio', 'Pasajeros presentes', 'Ingresos', 'Costo sobreventa', 'Utilidad',
            'Ingresos Acumulados', 'Costo Acumulado', 'Utilidad Acumulada',
            'Ingresos Promedio', 'Costo Promedio', 'Utilidad Promedio'
            ])
        self.run_simulation()
        
    def update_probabilities(self, probabilities: ConcurrenceProbabilities):
        self.probabilities = probabilities
        
        self.total_revenue = 0
        self.total_cost = 0
        self.total_profit = 0
        self.simulations = pd.DataFrame(columns=[
            'Vuelo', 'Número Aleatorio', 'Pasajeros presentes', 'Ingresos', 'Costo sobreventa', 'Utilidad',
            'Ingresos Acumulados', 'Costo Acumulado', 'Utilidad Acumulada',
            'Ingresos Promedio', 'Costo Promedio', 'Utilidad Promedio'
            ])
        self.run_simulation()
        
    def to_model(self) -> PandasModel:
        return PandasModel(self.simulations)