import numpy as np
import matplotlib.pyplot as plt
from NeuronalNetwork import NeuronalNetwork

#lambdas = [1, 1e-6, 10, 0.1, 0.01]
lambdas = [0.001, 0.0001, 0.00001]

results = []

for lambda_value in lambdas:
    neuronalNet = NeuronalNetwork('datasets/A233392.csv', skiprows=0)
    print(f"\n--- Entrenando Regresión Lambda = {lambda_value} ---")
    results.append(neuronalNet.train(epochs=1000, lambda_value=lambda_value))
    #neuronalNet = NeuronalNetwork('datasets/A233392.csv')
    #print(f"\n--- Iniciando Entrenamiento Lambda = {lambda_value} ---")
    #results.append(neuronalNet.train(epochs=500, lambda_value=lambda_value))


plt.figure(figsize=(12, 8))
# Solo graficamos los 3 resultados que existen
plt.plot(results[0].errors, label='Lambda = 0.1', color='blue')
plt.plot(results[1].errors, label='Lambda = 0.01', color='green')
plt.plot(results[2].errors, label='Lambda = 0.001', color='red')

plt.xlabel('Épocas')
plt.ylabel('Error')
plt.title('Evolución del Error (Regresión)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("results/evolucion_error_lambdas.png")

# Guardamos el reporte del primer experimento (índice 0)
results[0].plot_weights_curve("results/evolucion_pesos.png")
results[0].save_report_to_csv("results/reporte_red_neuronal.csv")



#class NeuronalNetwork:
    #def __init__(self, path_dataset: str, skiprows = 0):
        # 1. Cargar datos crudos
     #   raw_data = np.loadtxt(path_dataset, delimiter=",", skiprows=skiprows)

        # 2. NORMALIZAR (Obligatorio para evitar el overflow)
        # Esto convierte los valores de miles (0-7500) a un rango de 0 a 1
      #  self.dataset = (raw_data - raw_data.min(axis=0)) / (raw_data.max(axis=0) - raw_data.min(axis=0))

        # 3. Extraer valores ya normalizados
       # self.Y_values = self.__slicing_Y_values()
        #self.X_values = self.__slicing_X_values()
        #self.W_values = self.__generate_W_values()
