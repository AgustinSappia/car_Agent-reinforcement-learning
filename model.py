import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearQNet(nn.Module):
    """
    Red neuronal Q simplefgv
    - Entrada: tamaño del espacio de observación (e.g., 7: 5 sensores + speed + angle).
    - Capas ocultas: 128 neuronas (ReLU) -> 64 neuronas (ReLU).
    - Salida: número de acciones posibles (e.g., 4 acciones discretas).
    Esta red estima los valores Q(s, a) para cada acción.
    """
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=4):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)  # Salida lineal: valores Q para cada acción
        return x

    def save(self, file_name):
        """
        Guarda los pesos de la red en un archivo.
        """
        torch.save(self.state_dict(), file_name)
        print(f"Modelo guardado en {file_name}")