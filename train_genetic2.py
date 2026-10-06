"""
Atajo para el entrenamiento genético (se mantiene el nombre de siempre).

El entrenador viejo (una red de PyTorch por auto, todo en Python) se reemplazó
por el de la carpeta ai/, que simula toda la población junta con numpy.
Ver train.py.
"""

from train import main

if __name__ == "__main__":
    main()
