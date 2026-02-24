import numpy as np
import numpy.linalg as la

from entities.data_results import DataResults

class NeuronalNetwork:
    def __init__(self, path_dataset: str, skiprows=0):
        # 1. Carga única de datos
        raw_data = np.loadtxt(path_dataset, delimiter=",", skiprows=skiprows)

        # 2. Normalización Min-Max (Esto escala los 150 registros entre 0 y 1)
        # Es vital para que la regresión no explote con valores de 7000+
        self.dataset = (raw_data - raw_data.min(axis=0)) / (raw_data.max(axis=0) - raw_data.min(axis=0))

        # 3. Extraer valores del dataset YA normalizado
        self.Y_values = self.__slicing_Y_values()
        self.X_values = self.__slicing_X_values()
        self.W_values = self.__generate_W_values()

    def __slicing_Y_values(self):
        Y_values = self.dataset[:, -1]
        Y_values = Y_values.reshape(-1, 1)
        return Y_values
    
    def __slicing_X_values(self):
        X_values = self.dataset[:, :-1]
        m = X_values.shape[0]
        colum_ones = np.ones((m,1))
        X_values = np.hstack((colum_ones, X_values))
        return X_values
    
    def __generate_W_values(self):
        m = self.X_values.shape[1]
        W_values = np.random.randn(m, 1)
        return W_values
    
    def show_data_init(self):
        print(f"Dataset Inicial: \n{self.dataset.shape}")
        print(f"\nValores Y: \n{self.Y_values.shape}")
        print(f"\nValores X: \n{self.X_values.shape}")
        print(f"\nValores W: \n{self.W_values.shape}")
    
    def train(self, epochs: int, lambda_value: float) -> DataResults:

        history_evolution = DataResults(errors=[], weights=[])

        for i in range(epochs):
            y_c = self.__funtion_activation(self.X_values @ self.W_values)

            error = y_c - self.Y_values
            delta_w = - (lambda_value * (self.X_values.T @ error))
            self.W_values += delta_w

            
            error_magnitude = la.norm(error)

            history_evolution.errors.append(error_magnitude)
            history_evolution.weights.append(self.W_values.copy())
        
        return history_evolution
            
    def __funtion_activation(self, u):
        # np.where funciona así: np.where(condición, valor_si_verdad, valor_si_falso)
        return u
 