"""
Modelo Lineal Bayesiano
=======================

Implementación de regresión lineal Bayesiana con funciones base polinomiales.

Este módulo provee funciones para:
- Calcular el posterior de los pesos dado los datos
- Evaluar la verosimilitud de los datos
- Hacer predicciones con incertidumbre
- Calcular la evidencia del modelo (para selección de modelos)

Referencia: Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Cap. 3.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal as ruido_gaussiano
from scipy.stats import multivariate_normal as gaussiana_multivariada
from scipy.stats import norm as gaussiana_univariada

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

# Semillas para reproducibilidad
random.seed(1)
np.random.seed(1)

# Paleta de colores para gráficos
CMAP = plt.get_cmap("tab10")

# =============================================================================
# HIPERPARÁMETROS POR DEFECTO
# =============================================================================

# Precisión del prior sobre los pesos (inverso de la varianza)
# Un valor pequeño indica un prior poco informativo (alta varianza)
ALPHA: float = 10e-6  # Bishop usa alpha = 5e-3

# Precisión del ruido en los datos (inverso de la varianza del ruido)
# BETA = (1/sigma)^2 donde sigma es la desviación estándar del ruido
BETA: float = (1 / 0.2) ** 2


# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================


def posterior(
    t: np.ndarray, Phi: Matriz, alpha: float = ALPHA, beta: float = BETA
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la distribución posterior de los pesos del modelo lineal.

    Dado un prior Gaussiano sobre los pesos w ~ N(0, alpha^-1 * I) y
    un modelo de verosimilitud t ~ N(Phi @ w, beta^-1 * I), calcula
    el posterior p(w|t, Phi) = N(m_N, S_N).

    Fórmulas (Bishop ecuaciones 3.53-3.54):
        S_N^{-1} = alpha * I + beta * Phi^T @ Phi
        m_N = beta * S_N @ Phi^T @ t

    Args:
        t: Vector de valores objetivo (N,) o (N, 1).
        Phi: Matriz de diseño (N, M) donde N es el número de datos
             y M es el número de funciones base.
        alpha: Precisión del prior sobre los pesos.
        beta: Precisión del ruido en los datos.

    Returns:
        m_N: Media del posterior (M,) o (M, 1).
        S_N: Covarianza del posterior (M, M).

    Ejemplo:
        >>> X = np.array([[1, 0.5], [1, 1.0], [1, 1.5]])  # Phi con intercepto
        >>> t = np.array([0.8, 1.1, 1.4])
        >>> media, covarianza = posterior(t, X)
    """
    M = Phi.shape[1]  # Número de funciones base

    # Covarianza del posterior: S_N = (alpha*I + beta*Phi^T*Phi)^{-1}
    S_N_inv = alpha * np.eye(M) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)

    # Media del posterior: m_N = beta * S_N * Phi^T * t
    m_N = beta * S_N.dot(Phi.T).dot(t)

    return m_N, S_N


def likelihood(
    w: np.ndarray, t: np.ndarray, Phi: np.ndarray, beta: float = BETA
) -> float:
    """
    Calcula la verosimilitud de los datos dado un vector de pesos específico.

    Evalúa p(t|w, Phi, beta) = prod_i N(t_i | w^T @ phi_i, beta^{-1})

    Args:
        w: Vector de pesos del modelo (M,).
        t: Vector de valores objetivo (N,).
        Phi: Matriz de diseño (N, M).
        beta: Precisión del ruido.

    Returns:
        Verosimilitud (producto de las densidades evaluadas).

    Nota:
        Para estabilidad numérica, considerar usar log_likelihood en su lugar.
    """
    resultado = 1.0
    sigma = np.sqrt(beta ** (-1))

    for i in range(len(t)):
        media = w.T.dot(Phi[i])
        resultado = resultado * gaussiana_univariada.pdf(t[i], media, sigma)

    return resultado


def moments_predictive(
    Phi_posteriori: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
    t_priori: Optional[np.ndarray] = None,
    Phi_priori: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula los momentos (media y covarianza) de la distribución predictiva.

    La distribución predictiva p(t*|x*, D) es Gaussiana con:
        media = Phi* @ m_N
        varianza = 1/beta + Phi* @ S_N @ Phi*^T

    donde m_N y S_N son la media y covarianza del posterior calculados
    a partir de los datos priori.

    Args:
        Phi_posteriori: Matriz de diseño para los puntos a predecir (N*, M).
        alpha: Precisión del prior sobre los pesos.
        beta: Precisión del ruido.
        t_priori: Valores objetivo de entrenamiento (N,). Si es None, usa prior.
        Phi_priori: Matriz de diseño de entrenamiento (N, M). Si es None, usa prior.

    Returns:
        mu: Media de la predictiva (N*,) o (N*, 1).
        sigma2: Covarianza de la predictiva (N*, N*).
    """
    _, D = Phi_posteriori.shape

    # Si no hay datos previos, usar matrices vacías (solo el prior)
    if t_priori is None:
        t_priori = np.zeros((0, 1))
        Phi_priori = np.zeros((0, D))

    # Obtener posterior dado los datos de entrenamiento
    m_prior, S_prior = posterior(t_priori, Phi_priori, alpha, beta)

    # Media predictiva: mu = Phi* @ m_N
    mu = Phi_posteriori.dot(m_prior)

    # Covarianza predictiva: sigma^2 = 1/beta + Phi* @ S_N @ Phi*^T
    sigma2 = Phi_posteriori.dot(S_prior.dot(Phi_posteriori.T)) + (1 / beta) * np.eye(
        Phi_posteriori.shape[0]
    )

    return mu, sigma2


def predictive(
    t_posteriori: np.ndarray,
    Phi_posteriori: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
    t_priori: Optional[np.ndarray] = None,
    Phi_priori: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Evalúa la densidad predictiva en puntos específicos.

    Calcula p(t*|Phi*, D) donde D son los datos de entrenamiento.

    Args:
        t_posteriori: Valores donde evaluar la densidad (N*,).
        Phi_posteriori: Matriz de diseño para los puntos a predecir (N*, M).
        alpha: Precisión del prior sobre los pesos.
        beta: Precisión del ruido.
        t_priori: Valores objetivo de entrenamiento.
        Phi_priori: Matriz de diseño de entrenamiento.

    Returns:
        Densidad de probabilidad evaluada en t_posteriori.
    """
    m, S = moments_predictive(Phi_posteriori, alpha, beta, t_priori, Phi_priori)
    return gaussiana_multivariada.pdf(t_posteriori.ravel(), m.ravel(), S)


def log_evidence(
    t: np.ndarray, Phi: np.ndarray, alpha: float = ALPHA, beta: float = BETA
) -> np.ndarray:
    """
    Calcula el logaritmo de la evidencia del modelo (verosimilitud marginal).

    La evidencia p(t|alpha, beta) integra sobre todos los posibles pesos:
        p(t|alpha, beta) = integral p(t|w, beta) p(w|alpha) dw

    Esta cantidad es fundamental para la selección Bayesiana de modelos,
    ya que incorpora automáticamente un balance entre ajuste y complejidad
    (navaja de Occam Bayesiana).

    Fórmula (Bishop ecuación 3.86):
        ln p(t|alpha, beta) = (M/2) ln(alpha) + (N/2) ln(beta)
                              - E(m_N) - (1/2) ln|A| - (N/2) ln(2*pi)

    donde:
        E(m_N) = (beta/2)||t - Phi @ m_N||^2 + (alpha/2)||m_N||^2
        A = S_N^{-1}

    Args:
        t: Vector de valores objetivo (N,).
        Phi: Matriz de diseño (N, M).
        alpha: Precisión del prior.
        beta: Precisión del ruido.

    Returns:
        Log-evidencia del modelo (escalar, como array).
    """
    N, M = Phi.shape

    # Calcular posterior
    m_N, S_N = posterior(t, Phi, alpha, beta)

    # A = S_N^{-1} (matriz de precisión del posterior)
    A = np.linalg.inv(S_N)
    A_det = np.linalg.det(A)

    # Término de energía evaluado en la media del posterior
    # E(m_N) = (beta/2)||t - Phi*m_N||^2 + (alpha/2)||m_N||^2
    residuo = t - Phi.dot(m_N)
    E_mN = (beta / 2) * residuo.T.dot(residuo) + (alpha / 2) * m_N.T.dot(m_N)

    # Log-evidencia (Bishop eq. 3.86)
    log_evidencia = (
        (M / 2) * np.log(alpha)
        + (N / 2) * np.log(beta)
        - E_mN
        - (1 / 2) * np.log(A_det)
        - (N / 2) * np.log(2 * np.pi)
    )

    return log_evidencia


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================


def sinus_model(X: np.ndarray, variance: float) -> np.ndarray:
    """
    Genera datos de una función seno con ruido gaussiano.

    Modelo: y = sin(2*pi*x) + epsilon, donde epsilon ~ N(0, variance)

    Args:
        X: Valores de entrada.
        variance: Varianza del ruido gaussiano.

    Returns:
        Valores de salida con ruido añadido.
    """
    return np.sin(2 * np.pi * X) + ruido_gaussiano(0, np.sqrt(variance), X.shape)


def polynomial_basis_function(x: np.ndarray, degree: int = 1) -> np.ndarray:
    """
    Función base polinomial.

    Transforma x en x^degree para construir la matriz de diseño.

    Args:
        x: Valores de entrada.
        degree: Grado del polinomio.

    Returns:
        x elevado a la potencia indicada.

    Ejemplo:
        >>> x = np.array([1, 2, 3])
        >>> polynomial_basis_function(x, degree=2)
        array([1, 4, 9])
    """
    return x**degree
