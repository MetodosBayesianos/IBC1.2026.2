import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# Datos: cantidad de positivos (0 a 3) de cada operador (fila) en cada contexto (columna)
data = pd.read_csv("../2-data/positivos_por_cepas_verdaderos.csv")
contextos = ["20par-DTU1", "20par-DTU2", "20par-DTU6", "50par-DTU1", "50par-DTU2", "50par-DTU6"]
N = 3
n = np.round(data[contextos].to_numpy() / 100 * N).astype(int)


# Modelos
# Todos comparten el desempeño de cada contexto, p_c ~ Beta(2,2).
# Cada función crea los parámetros propios del modelo y devuelve log P(Datos | parámetros).

def base(p):
    # n_ci ~ Binomial(3, p_c)
    return pm.logp(pm.Binomial.dist(n=N, p=p), n).sum()

def operadores(p):
    # q ~ Beta(1,9): probabilidad de que un operador trabaje mal
    # r ~ Beta(1,1): factor que reduce el desempeño cuando el operador trabaja mal
    # C_i ~ Bernoulli(q), sumado sobre sus dos valores:
    #   P(n_i) = (1-q) prod_c Binomial(n_ci | 3, p_c) + q prod_c Binomial(n_ci | 3, p_c r)
    q = pm.Beta("q", alpha=1, beta=9)
    r = pm.Beta("r", alpha=1, beta=1)
    log_bien = pt.log(1 - q) + pm.logp(pm.Binomial.dist(n=N, p=p), n).sum(axis=1)
    log_mal = pt.log(q) + pm.logp(pm.Binomial.dist(n=N, p=p * r), n).sum(axis=1)
    return pt.logaddexp(log_bien, log_mal).sum()

modelos = {"Base": base, "Operadores": operadores}  # Para agregar un modelo, sumarlo acá


# Selector de modelos
# M es una variable que elige qué modelo genera los datos. Por la regla de Bayes,
#   P(M | Datos) ∝ P(Datos | M) P(M)
# así que la evidencia de cada modelo es, salvo una constante, P(M | Datos) / P(M).

def selector(prior_M, muestras=10000):
    with pm.Model():
        p = pm.Beta("p", alpha=2, beta=2, shape=len(contextos))
        log_verosimilitudes = pt.stack([modelo(p) for modelo in modelos.values()])
        M = pm.Categorical("M", p=prior_M)
        pm.Potential("verosimilitud", log_verosimilitudes[M])
        traza = pm.sample(muestras, tune=2000, chains=4, target_accept=0.95)
    M = traza.posterior["M"].values.ravel()
    return np.array([np.mean(M == k) for k in range(len(modelos))])

# Si un modelo es mucho mejor que otro, el selector casi nunca visita al peor y la
# estimación es mala. Por eso primero lo corremos con un prior uniforme y después
# le damos más prior a los modelos poco visitados, hasta que los visite a todos.
prior_M = np.ones(len(modelos)) / len(modelos)
for intento in range(3):
    posterior_M = selector(prior_M)
    log_evidencia = np.log(np.maximum(posterior_M, 1e-4)) - np.log(prior_M)  # salvo una constante
    print(f"\nPrior de M: {np.round(prior_M, 4)}   Posterior de M: {np.round(posterior_M, 4)}")
    if posterior_M.min() > 0.05:
        break
    prior_M = np.exp(-log_evidencia) / np.exp(-log_evidencia).sum()


# Resultados
log_evidencia -= log_evidencia.max()
print("\nlog evidencia relativa al mejor modelo:")
for nombre, le in zip(modelos, log_evidencia):
    print(f"  {nombre:12s} {le:8.2f}")
print("\nP(Modelo | Datos) con prior uniforme entre modelos:")
for nombre, pm_ in zip(modelos, np.exp(log_evidencia) / np.exp(log_evidencia).sum()):
    print(f"  {nombre:12s} {pm_:.4f}")
