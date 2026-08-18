"""
Práctica 1.2: Argumentos causales alternativos e incertidumbre.
============================================
"""

import warnings
from typing import Callable, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Espacio de estados
# ------------------------------------------------------------

# Cajas posibles {0, 1, 2}
H = np.arange(3)

# Dominio discreto para aproximar la integral en p
P_GRID = np.linspace(0.0, 1.0, 21)


# ------------------------------------------------------------
# 2. Definición de métodos de la práctica anterior que
#    se utilizan en esta práctica
# ------------------------------------------------------------


def p_r(r: int) -> float:
    """
    P(R = r).
    Prior sobre la ubicación del regalo.
    r ∈ {0, 1, 2}
    """
    return 1 / 3


def p_c(c: int) -> float:
    """
    P(C = c).
    Prior sobre la caja elegida por el participante
    c ∈ {0, 1, 2}
    """
    return 1 / 3


def p_s_rM0(s: int, r: int) -> float:
    """
    P(S = s | R= r, M = 0).
    Modelo 0 (No Monty Hall):
    El presentador abre cualquier caja que no tenga el regalo.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    """
    if s != r:
        return 1 / 2
    return 0


def p_s_rcM1(s: int, r: int, c: int):
    """
    P(S = s | R= r, C= c, M = 1).
    Modelo 1 (Monty Hall):
    El presentador abre una caja que no tenga el regalo
    ni haya sido seleccionada.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    c ∈ {0, 1, 2}
    """
    if not (s != r and s != c):
        return 0

    cajas_validas = [h for h in H if h != r and h != c]
    return 1 / len(cajas_validas)


def p_rcs_M(r: int, c: int, s: int, m: int) -> float:
    """
    P(r, c, s | M) = P(r | M)P(c | M)P(s | r, c, M)
    Distribución conjunta del modelo m.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    c ∈ {0, 1, 2}
    m ∈ {0, 1}
    """
    base = p_r(r) * p_c(c)

    if m == 0:
        return base * p_s_rM0(s, r)
    else:
        return base * p_s_rcM1(s, r, c)


def pDatos_M(datos: List[Tuple[int, int, int]], m: int) -> float:
    """
    P(Datos | M)
    Probabilidad de ver los datos dados el modelo considerado.
    m ∈ {0, 1}
    """
    prob = 1
    for c, s, r in datos:
        prob *= p_rcs_M(r, c, s, m)
    return prob


def log_pDatos_M(datos: List[Tuple[int, int, int]], m: int) -> float:
    """
    log_10(P(Datos | M))
    Logaritmo de la probabilidad de ver los datos dados el modelo considerado.
    m ∈ {0, 1}
    """
    prob = 0
    for c, s, r in datos:
        valor = p_rcs_M(r, c, s, m)
        if valor > 0:
            prob += np.log10(valor)
        else:
            return -np.inf
    return prob


# ------------------------------------------------------------
# 3. Priors sobre la memoria p y la probabilidad de acordarse a
# ------------------------------------------------------------


def p_p(p: float) -> float:
    """
    P(P = p).
    Prior sobre la memoria del presentador.
    p ∈ [0, 1]
    """
    pass


def p_a_p(a: int, p: float) -> float:
    """
    P(A = a | p)
    Probabilidad de que el presentador recuerde o no la elección
    del participante.
    a ∈ {0, 1}
    p ∈ [0, 1]
    """
    pass


# ------------------------------------------------------------
# 4. Verosimilitud de un episodio dada la memoria p
# ------------------------------------------------------------


def p_csr_p(r: int, c: int, s: int, p: float) -> float:
    """
    P(r, c, s | p) = p * P(r, c, s | M = M1) + (1 - p) * P(r, c, s | M = M0)
    Verosimilitud de un episodio dado la memoria p.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    c ∈ {0, 1, 2}
    p ∈ [0, 1]
    """
    pass


# ------------------------------------------------------------
# 5. Posterior de la memoria p
# ------------------------------------------------------------


def p_p_datos(p: float, datos: List[Tuple[int, int, int]]) -> float:
    """
    P(p | (r0,  c0, s0), (r1,  c1, s1), ...)
    Distribución posterior sobre la memoria p dados los datos observados.
    Datos: Lista de tuplas (c, s, r)
    p ∈ [0, 1]
    """
    pass


# ------------------------------------------------------------
# 6. Estimación de p mediante datos observados
# ------------------------------------------------------------


def estimar_p(posterior_p: Callable, observaciones: int = 60) -> pd.DataFrame:
    """
    Función de ayuda que permite estimar la memoria p mediante datos observados.
    posterior_p: Función que calcula la posterior de p dado los datos observados.
    observaciones: Número de observaciones a considerar.
    """

    no_monty = pd.read_csv("NoMontyHall.csv")
    no_monty = list(no_monty.itertuples(index=False, name=None))
    no_monty = no_monty[:observaciones]

    estimaciones_p = pd.DataFrame(columns=["p", "posterior"])

    for p in P_GRID:
        estimaciones_p.loc[len(estimaciones_p)] = [p, posterior_p(p, no_monty)]

    return estimaciones_p


# ------------------------------------------------------------
# 7. Predicción de un episodio considerando datos anteriores
# ------------------------------------------------------------


def p_episodio_datos_MA(
    r: int, c: int, s: int, datos: List[Tuple[int, int, int]]
) -> float:
    """
    P(r, c, s | datos_0:T-1, MA)
    Probabilidad de ver un episodio dado los datos anteriores considerando
    al modelo alternativo.
    s ∈ {0, 1, 2}
    r ∈ {0, 1, 2}
    c ∈ {0, 1, 2}
    Datos: Lista de tuplas (c, s, r)
    """
    pass


# ------------------------------------------------------------
# 8. Predicción del modelo alternativo sobre todo el
#    conjunto de datos observado
# ------------------------------------------------------------


def p_datos_MA(datos: List[Tuple[int, int, int]]) -> float:
    """
    P(Datos_0:T | MA)
    Probabilidad de ver los datos observados considerando el modelo alternativo.
    Se calcula como el producto de las predicciones secuenciales:
    P(Ep_0|MA) * P(Ep_1|Datos_0, MA) * P(Ep_2|Datos_0:1, MA) * ...
    Datos: Lista de tuplas (c, s, r)

    NOTA PARA RESOLUCION:
    ---------------------

    En lugar de recalcular P(p | datos_0:t-1) desde cero en cada paso t
    (lo cual requiere recorrer todos los datos previos por cada valor de p),
    aprovechamos que el posterior se puede actualizar incrementalmente
    usando la regla de Bayes:

        P(p | datos_0:t) ∝ P(episodio_t | p) × P(p | datos_0:t-1)

    Así, mantenemos el posterior sobre p como una lista que vamos
    actualizando episodio a episodio: multiplicamos por la verosimilitud
    del nuevo episodio y renormalizamos.
    """
    pass


def log_p_datos_MA(datos: List[Tuple[int, int, int]]) -> float:
    """
    log_10(P(Datos_0:T | MA))
    Logaritmo de la probabilidad de ver los datos observados considerando
    el modelo alternativo. Se trabaja en escala logarítmica para evitar
    underflow numérico en datasets grandes.
    Datos: Lista de tuplas (c, s, r)

    NOTA PARA RESOLUCION:
    ---------------------

    Misma idea que p_datos_MA, pero acumulando en escala logarítmica:
    en vez de multiplicar las predicciones, sumamos sus logaritmos.

    En lugar de recalcular P(p | datos_0:t-1) desde cero en cada paso t,
    mantenemos el posterior sobre p y lo actualizamos incrementalmente:

        P(p | datos_0:t) ∝ P(episodio_t | p) × P(p | datos_0:t-1)
    """
    pass


# ------------------------------------------------------------
# 9. Cálculo del bayes factor, modelo alternativo vs base y
#    monty hall.
#   (diferencia en órdenes de magnitud en escala logarítmica)
# ------------------------------------------------------------


def log_bayes_factor(log_pdatos_mi: float, log_pdatos_mj: float) -> float:
    """
    log_10(BF_ij) = log_10(P(Datos | Mi)) - log_10(P(Datos | Mj))
    Bayes factor en escala logarítmica para comparar dos modelos.
    Recibe los logaritmos de las verosimilitudes ya calculados.
    El resultado se interpreta como órdenes de magnitud a favor de Mi.
    """
    pass


# ------------------------------------------------------------
# 10. Predicción típica del modelo alternativo sobre los datos
# ------------------------------------------------------------


def log_media_geometrica(datos: List[Tuple[int, int, int]], m: int) -> float:
    """
    log_10(Media Geométrica) = (1/N) * log_10(P(Datos | M))
    Logaritmo de la predicción típica (media geométrica) de los datos
    dado el modelo considerado.
    m ∈ {0, 1}
    """
    pass


def log_media_geometrica_MA(datos: List[Tuple[int, int, int]]) -> float:
    """
    log_10(Media Geométrica MA) = (1/N) * log_10(P(Datos | MA))
    Logaritmo de la predicción típica (media geométrica) de los datos
    dado el modelo alternativo.
    """
    pass


# ------------------------------------------------------------
# 11. Actualización de creencias considerando al modelo
#     alternativo (posterior con 3 modelos)
# ------------------------------------------------------------


def pM_con_alternativo(m: Literal[0, 1, "MA"]) -> float:
    """
    P(M)
    Prior uniforme sobre los 3 modelos (Base, Monty Hall, Alternativo).
    m ∈ {0, 1, 'MA'}
    """
    pass


def p_datos_con_alternativo(datos: List[Tuple[int, int, int]]) -> float:
    """
    P(Datos)
    Probabilidad total de ver los datos usando la contribución
    de los 3 modelos (Base, Monty Hall y Alternativo).
    """
    pass


def pM_datos_con_alternativo(
    m: Literal[0, 1, "MA"], datos: List[Tuple[int, int, int]]
) -> float:
    """
    P(M | Datos)
    Actualización de la creencia sobre el modelo M dado los datos
    (distribución a posteriori) considerando los 3 modelos.
    m ∈ {0, 1, 'MA'}
    """
    pass


def evolucion_posterior_con_alternativo(
    m: Literal[0, 1, "MA"], datos: List[Tuple[int, int, int]]
) -> List[float]:
    """
    Evolución del posterior del modelo m según se observa información,
    considerando los 3 modelos (Base, Monty Hall y Alternativo).
    m ∈ {0, 1, 'MA'}
    """
    pass


# ------------------------------------------------------------
# xx. Ejecución principal
# ------------------------------------------------------------

if __name__ == "__main__":
    # Carga de datos
    no_monty_df = pd.read_csv("NoMontyHall.csv")
    no_monty = list(no_monty_df.itertuples(index=False, name=None))

    # Estimaciones de p
    print(estimar_p(p_p_datos))

    # --- Log-probabilidades de los datos dado cada modelo ---
    log_pdatos_M0 = log_pDatos_M(no_monty, m=0)
    log_pdatos_M1 = log_pDatos_M(no_monty, m=1)
    log_pdatos_MA = log_p_datos_MA(no_monty)

    print(f"Logaritmo de P(datos|M0): {log_pdatos_M0:.5e}")
    print(f"Logaritmo de P(datos|M1): {log_pdatos_M1:.5e}")
    print(f"Logaritmo de P(datos|MA): {log_pdatos_MA:.5e}")
    print()

    # Verificación en escala normal
    print(f"P(datos|M0) ≈ 10^({log_pdatos_M0:.2f}) = {10**log_pdatos_M0:.5e}")
    print(f"P(datos|M1) ≈ 10^({log_pdatos_M1:.2f})")
    print(f"P(datos|MA) ≈ 10^({log_pdatos_MA:.2f}) = {10**log_pdatos_MA:.5e}")
    print()

    # --- Bayes Factor ---
    log_bf_M1_M0 = log_bayes_factor(log_pdatos_M1, log_pdatos_M0)
    log_bf_MA_M0 = log_bayes_factor(log_pdatos_MA, log_pdatos_M0)
    log_bf_MA_M1 = log_bayes_factor(log_pdatos_MA, log_pdatos_M1)

    print(f"Logaritmo de Bayes Factor M1/M0: {log_bf_M1_M0:.5e}")
    print(f"Logaritmo de Bayes Factor MA/M0: {log_bf_MA_M0:.5e}")
    print(f"Logaritmo de Bayes Factor MA/M1: {log_bf_MA_M1:.5e}")
    print()

    # --- Predicción típica (media geométrica) ---
    print(
        "Predicción típica de los datos por parte de M0: ",
        10 ** log_media_geometrica(no_monty, m=0),
    )
    print(
        "Predicción típica de los datos por parte de M1: ",
        10 ** log_media_geometrica(no_monty, m=1),
    )
    print(
        "Predicción típica de los datos por parte de MA: ",
        10 ** log_media_geometrica_MA(no_monty),
    )
    print()

    # --- Gráfico de evolución del posterior con 3 modelos ---
    datos_grafico = no_monty[:60]

    post_M0 = evolucion_posterior_con_alternativo(m=0, datos=datos_grafico)
    post_M1 = evolucion_posterior_con_alternativo(m=1, datos=datos_grafico)
    post_MA = evolucion_posterior_con_alternativo(m="MA", datos=datos_grafico)

    plt.figure(figsize=(8, 6))
    plt.plot(post_M0, label="M0: Base")
    plt.plot(post_M1, label="M1: Monty Hall")
    plt.plot(post_MA, label="MA: Modelo Alternativo")
    plt.xlabel("Número de episodios")
    plt.ylabel("P(Modelo | Datos)")
    plt.title("Evolución del posterior de los modelos")
    plt.legend()
    plt.tight_layout()
    plt.show()
