"""
Cerebros de toda la población en arrays de numpy.

En vez de tener una red de PyTorch por auto (y evaluarlas de a una), los pesos de
todos los autos se guardan juntos: W[capa] tiene forma (población, entradas, salidas).
Así una sola multiplicación de matrices calcula la decisión de todos los autos a la vez.
"""

import numpy as np


class PopulationBrain:
    def __init__(self, layer_sizes, population, rng=None):
        self.sizes = list(layer_sizes)
        self.population = population
        self.rng = rng or np.random.default_rng()
        self.W = []
        self.b = []
        for n_in, n_out in zip(self.sizes, self.sizes[1:]):
            self.W.append(self.rng.normal(0, 1 / np.sqrt(n_in), (population, n_in, n_out)).astype(np.float32))
            self.b.append(np.zeros((population, n_out), dtype=np.float32))

    # ------------------------------------------------------------------ #
    def forward(self, x, idx=None, return_activations=False):
        """
        x: (n, entradas). idx: índices de los autos (por defecto todos).
        Devuelve la acción elegida por cada auto (y opcionalmente las activaciones).
        """
        W = self.W if idx is None else [w[idx] for w in self.W]
        b = self.b if idx is None else [v[idx] for v in self.b]
        h = x.astype(np.float32)
        acts = [h]
        for k, (w, bias) in enumerate(zip(W, b)):
            h = np.matmul(h[:, None, :], w)[:, 0, :] + bias
            if k < len(W) - 1:
                h = np.tanh(h)
            acts.append(h)
        choice = np.argmax(h, axis=1)
        return (choice, acts) if return_activations else choice

    # ------------------------------------------------------------------ #
    def get(self, i):
        """Pesos de un auto (para guardar o copiar)"""
        return [w[i].copy() for w in self.W], [v[i].copy() for v in self.b]

    def set(self, i, genome):
        W, b = genome
        for k in range(len(self.W)):
            self.W[k][i] = W[k]
            self.b[k][i] = b[k]

    def evolve(self, fitness, elite_pct=20, mutation_rate=0.1, mutation_strength=0.3, crossover=True):
        """
        Nueva generación:
        - Los mejores (élite) pasan sin cambios
        - El resto son hijos de padres elegidos por torneo entre la mejor mitad,
          mezclando neuronas de dos padres (cruza) y con mutaciones aleatorias
        """
        P = self.population
        order = np.argsort(-np.asarray(fitness))
        n_elite = max(1, int(P * elite_pct / 100))
        pool = order[:max(2, P // 2)]

        def pick():
            a, b = self.rng.choice(pool, 2, replace=len(pool) < 2)
            return a if fitness[a] >= fitness[b] else b

        newW = [np.empty_like(w) for w in self.W]
        newb = [np.empty_like(v) for v in self.b]
        for slot in range(P):
            if slot < n_elite:
                src = order[slot]
                for k in range(len(self.W)):
                    newW[k][slot] = self.W[k][src]
                    newb[k][slot] = self.b[k][src]
                continue
            pa = pick()
            pb = pick() if crossover else pa
            for k in range(len(self.W)):
                if crossover and pa != pb:
                    # Cada neurona de salida viene entera de uno de los dos padres
                    mask = self.rng.random(self.W[k].shape[2]) < 0.5
                    newW[k][slot] = np.where(mask[None, :], self.W[k][pa], self.W[k][pb])
                    newb[k][slot] = np.where(mask, self.b[k][pa], self.b[k][pb])
                else:
                    newW[k][slot] = self.W[k][pa]
                    newb[k][slot] = self.b[k][pa]

        # Mutación (no a la élite)
        for arrs in (newW, newb):
            for a in arrs:
                tail = a[n_elite:]
                hit = self.rng.random(tail.shape) < mutation_rate
                tail += hit * self.rng.normal(0, mutation_strength, tail.shape).astype(np.float32)
        self.W, self.b = newW, newb
        return order

    # ------------------------------------------------------------------ #
    def save_best(self, path, i, extra=None):
        W, b = self.get(i)
        data = {f"W{k}": w for k, w in enumerate(W)}
        data.update({f"b{k}": v for k, v in enumerate(b)})
        data['sizes'] = np.array(self.sizes)
        for key, value in (extra or {}).items():
            data[key] = np.array(value)
        np.savez(path, **data)

    def load_into_all(self, path, keep_fraction=0.1, mutation_strength=0.3):
        """
        Carga un cerebro guardado. Una parte de la población arranca como copia exacta
        y el resto como variaciones, para seguir mejorándolo.
        Devuelve False si el cerebro no es compatible (otro tamaño de red).
        """
        data = np.load(path)
        if list(data['sizes']) != self.sizes:
            return False
        W = [data[f"W{k}"] for k in range(len(self.W))]
        b = [data[f"b{k}"] for k in range(len(self.b))]
        n_keep = max(1, int(self.population * keep_fraction))
        for i in range(self.population):
            self.set(i, (W, b))
            if i >= n_keep:
                for k in range(len(self.W)):
                    self.W[k][i] += self.rng.normal(0, mutation_strength, W[k].shape).astype(np.float32)
                    self.b[k][i] += self.rng.normal(0, mutation_strength, b[k].shape).astype(np.float32)
        return True
