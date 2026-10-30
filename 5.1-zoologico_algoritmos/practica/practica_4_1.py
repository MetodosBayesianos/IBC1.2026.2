"""Práctica 4.1: positividad y competencias (plantilla para completar)."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import pandas as pd
from algoritmos import ALGORITMOS, NOMBRES, propension

# las redes quedan en algoritmos.py para explorar, pero no se corren: duplicarian el tiempo de la corrida
ALGORITMOS = {nombre: a for nombre, a in ALGORITMOS.items() if nombre not in ("tarnet", "dragonnet")}

AQUI = Path(__file__).parent
DATOS = AQUI / "datos" / "data_cf_all"
K_DATOS = (0.05, 0.2, 1)
K_EXPERIMENTO = (0.05, 0.1, 0.2, 0.5, 1)
ESCENARIOS = (4, 7)  # ACIC 2016: iguales salvo el solapamiento (4 total, 7 limitado)
# con estos parametros, una replica tarda unos 20 s en Hill y unos 165 s en ACIC
PARAMETROS = {"matching": dict(n_boot=200), "ipw": dict(n_boot=50),
              "s_learner": dict(n_boot=20), "t_learner": dict(n_boot=20),
              "bart": dict(n_burnin=500, n_mcmc=500), "bcf": dict(n_burnin=500, n_mcmc=500)}
N_REPLICAS_HILL = 10
N_REPLICAS_ACIC = 10

plt.rcParams.update({"font.size": 14})
cmap = plt.get_cmap("tab10")
TITULOS = {"error_ate": "Error en el ATE", "pehe": "PEHE", "cubre": "Cobertura"}


def guardar(fig, nombre):
    """Figura en PDF (para la consigna) y en PNG (para el markdown de la entrega)."""
    fig.savefig(AQUI / f"{nombre}.pdf")
    fig.savefig(AQUI / f"{nombre}.png", dpi=150)


# 1.1 El modelo

def mu0(q):
    return 72 + 3 * np.sqrt(np.abs(q))


def mu1(q):
    return 90 + np.exp(0.06 * q)


def tau(q):
    return mu1(q) - mu0(q)


def zona_solapamiento(q, t):
    """Intervalo de q donde hay datos de los dos grupos."""
    return max(q[t == 0].min(), q[t == 1].min()), min(q[t == 0].max(), q[t == 1].max())


def asignacion(q, k):
    """Probabilidad de tratamiento P(T=1|q) = 1 / (1 + exp(-k (q - 30))), para un array q."""
    pass


def simular(k, semilla, n=120):
    """Simulacion de Hill con positividad controlada por k; devuelve los arrays q, t, y.

    Q: mezcla de N(20, 10^2) y N(40, 10^2) con pesos 1/2.
    T: Bernoulli con probabilidad asignacion(q, k).
    Y: normal con media mu0(q) si t = 0 o mu1(q) si t = 1, y desvio 1.
    t es un array de enteros 0 y 1.
    Usar rng = np.random.default_rng(semilla) para todo lo aleatorio.
    El corrector solo mira la distribucion; para reproducir exactamente las figuras de la consigna, sortear en este orden:
    la componente de la mezcla con rng.random(n) < 0.5, q con rng.normal, t con rng.binomial e y con rng.normal.
    """
    pass


def figura_positividad_datos():
    grilla = np.linspace(-13, 75, 200)
    fig, axes = plt.subplots(1, len(K_DATOS), figsize=(15, 4.5), sharey=True)
    for ax, k in zip(axes, K_DATOS):
        q, t, y = simular(k, 0)
        inferior, superior = zona_solapamiento(q, t)
        ax.axvspan(grilla[0], inferior, color="gray", alpha=0.15, lw=0)
        ax.axvspan(superior, grilla[-1], color="gray", alpha=0.15, lw=0)
        ax.scatter(q[t == 0], y[t == 0], color=cmap(0), alpha=0.6, label="Controles")
        ax.scatter(q[t == 1], y[t == 1], color=cmap(1), alpha=0.6, marker="^", label="Tratados")
        ax.plot(grilla, mu0(grilla), color=cmap(0), lw=2, label="$\\mu_0$ verdadera")
        ax.plot(grilla, mu1(grilla), color=cmap(1), lw=2, label="$\\mu_1$ verdadera")
        ax.set_xlim(grilla[0], grilla[-1])
        ax.set_ylim(70, 135)
        ax.set_title(f"$k = {k}$")
        ax.set_xlabel("$Q$")
    axes[0].set_ylabel("$Y$")
    axes[0].legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    guardar(fig, "positividad_datos")


# 1.2 El zoologico en cada k

def correr(X, t, y, semilla):
    """Todos los algoritmos del zoologico sobre los mismos datos."""
    resultados = {}
    for nombre, algoritmo in ALGORITMOS.items():
        semillas = {} if nombre == "ingenua" else {"semilla": semilla}
        resultados[nombre] = algoritmo(X, t, y, **PARAMETROS.get(nombre, {}), **semillas)
    return resultados


def metricas(r, tau_verdadero):
    """Error en el ATE, PEHE y si el intervalo del ATE cubre al verdadero.

    r: resultado de un algoritmo de algoritmos.py (claves "ate", "ate_inf", "ate_sup" y "tau", que puede ser None).
    tau_verdadero: efecto verdadero de cada individuo; el ATE verdadero es tau_verdadero.mean().
    Devuelve {"error_ate": |ate - ATE|, "pehe": raiz del promedio de (tau_estimado - tau_verdadero)^2,
    "cubre": ate_inf <= ATE <= ate_sup}, con tau_estimado igual al ate repetido si r["tau"] es None.
    Si el algoritmo no da estimacion (ate es nan), error_ate y pehe quedan nan y cubre es False.
    """
    pass


def experimento_positividad():
    """Metricas de todos los algoritmos en cada k y replica de Hill, en positividad.csv."""
    filas = []
    for k in K_EXPERIMENTO:
        for replica in range(N_REPLICAS_HILL):
            q, t, y = simular(k, replica)
            for nombre, r in correr(q[:, None], t, y, replica).items():
                filas.append(dict(k=k, replica=replica, algoritmo=nombre, **metricas(r, tau(q))))
            print("k", k, "replica", replica, flush=True)
            pd.DataFrame(filas).to_csv(AQUI / "positividad.csv", index=False)


# 1.3 Resultados

def medias(archivo, grupos):
    """Promedio de las metricas sobre las replicas, sin saltear nan.

    Un algoritmo sin estimacion en alguna replica queda sin punto de error y PEHE, y esa replica cuenta como no cubierta.
    """
    return pd.read_csv(AQUI / archivo).groupby(grupos)[list(TITULOS)].mean(skipna=False)


def figura_positividad_metricas():
    tabla = medias("positividad.csv", ["algoritmo", "k"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    for ax, (columna, titulo) in zip(axes, TITULOS.items()):
        for i, nombre in enumerate(ALGORITMOS):
            ax.plot(K_EXPERIMENTO, tabla.loc[nombre, columna], marker="o", color=cmap(i), label=NOMBRES[nombre])
        ax.set_xscale("log")
        ax.set_xlabel("$k$")
        ax.set_title(titulo)
    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[2].axhline(0.95, color="black", lw=1, ls="--")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=12)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    guardar(fig, "positividad_metricas")


# 2.1 Los datos

def cargar_acic(carpeta, escenario, replica):
    """Covariables, tratamiento, resultado observado y efecto verdadero de una replica de ACIC 2016.

    carpeta: ruta a data_cf_all (str o Path); escenario: numero de su subcarpeta.
    replica: indice desde 0 sobre los zymu_*.csv del escenario, ordenados por el numero del nombre (zymu_9 antes que zymu_10).
    Devuelve X, t, y, tau como arrays de numpy:
    X: pd.get_dummies(x, dtype=float) sobre x.csv, con las columnas que tienen algun valor fuera de {0, 1}
       estandarizadas con .mean() y .std() de pandas, convertido con .to_numpy();
       con los datos de ACIC queda de 4802 x 82 (las categoricas x_2, x_21 y x_24 pasan a indicadoras);
    t: la columna z; y: y1 donde z vale 1 e y0 donde no; tau: mu1 - mu0.
    """
    pass


# 2.2 Solapamiento con 58 covariables

def figura_acic_propension():
    fig, axes = plt.subplots(1, len(ESCENARIOS), figsize=(10, 4), sharey=True)
    for ax, escenario in zip(axes, ESCENARIOS):
        X, t, _, _ = cargar_acic(DATOS, escenario, 0)
        e, _ = propension(X, t)
        ax.hist(e[t == 0], bins=30, range=(0, 1), density=True, color=cmap(0), alpha=0.6, label="Controles")
        ax.hist(e[t == 1], bins=30, range=(0, 1), density=True, color=cmap(1), alpha=0.6, label="Tratados")
        ax.set_title(f"Escenario {escenario}")
        ax.set_xlabel("$\\hat e(x)$")
    axes[0].legend()
    fig.tight_layout()
    guardar(fig, "acic_propension")


# 2.3 El zoologico en los dos escenarios

def experimento_acic():
    """Metricas de todos los algoritmos en cada escenario y replica de ACIC 2016, en acic.csv.

    De la primera replica de cada escenario guarda ademas el efecto estimado de cada individuo, en acic_tau.csv.
    """
    filas, filas_tau = [], []
    for escenario in ESCENARIOS:
        for replica in range(N_REPLICAS_ACIC):
            X, t, y, tau_verdadero = cargar_acic(DATOS, escenario, replica)
            for nombre, r in correr(X, t, y, replica).items():
                filas.append(dict(escenario=escenario, replica=replica, algoritmo=nombre, **metricas(r, tau_verdadero)))
                if replica == 0:
                    tau_estimado = np.full(len(t), r["ate"]) if r["tau"] is None else r["tau"]
                    filas_tau.append(pd.DataFrame(dict(escenario=escenario, individuo=range(len(t)), algoritmo=nombre,
                                                       tau=tau_verdadero, tau_estimado=tau_estimado)))
            print("escenario", escenario, "replica", replica, flush=True)
            pd.DataFrame(filas).to_csv(AQUI / "acic.csv", index=False)
            pd.concat(filas_tau).to_csv(AQUI / "acic_tau.csv", index=False)


def figura_acic_metricas():
    tabla = medias("acic.csv", ["escenario", "algoritmo"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    posiciones = np.arange(len(ALGORITMOS))
    for ax, (columna, titulo) in zip(axes, TITULOS.items()):
        # cada escenario un poco corrido, para que no se tapen cuando coinciden
        for escenario, marcador, corrimiento in zip(ESCENARIOS, ("o", "s"), (-0.15, 0.15)):
            ax.plot(tabla.loc[escenario].loc[list(ALGORITMOS), columna], posiciones + corrimiento, marcador, ms=8,
                    label=f"Escenario {escenario}")
        ax.set_yticks(posiciones, [NOMBRES[n] if ax is axes[0] else "" for n in ALGORITMOS])
        ax.invert_yaxis()
        ax.set_title(titulo)
    axes[2].axvline(0.95, color="black", lw=1, ls="--")
    axes[2].legend(fontsize=10)
    fig.tight_layout()
    guardar(fig, "acic_metricas")


# 2.4 Estimado vs verdadero

def figura_acic_tau():
    """Efecto estimado contra verdadero de cada individuo en la primera replica, un panel por algoritmo."""
    tabla = pd.read_csv(AQUI / "acic_tau.csv")
    # unos pocos individuos con efectos de hasta 200 aplastarian al resto: quedan fuera de los ejes
    limites = tabla["tau"].quantile([0.005, 0.995]).to_numpy() + [-2, 2]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.5), sharex=True, sharey=True)
    for ax, nombre in zip(axes.flat, ALGORITMOS):
        ax.plot(limites, limites, color="black", lw=1)
        for escenario, color in zip(ESCENARIOS, (cmap(0), cmap(1))):
            d = tabla.query("escenario == @escenario and algoritmo == @nombre")
            ax.scatter(d["tau"], d["tau_estimado"], s=3, alpha=0.3, color=color, label=f"Escenario {escenario}")
        ax.set_title(NOMBRES[nombre], fontsize=14)
        ax.set_xlim(limites)
        ax.set_ylim(limites)
    fig.supxlabel("$\\tau_i$ verdadero")
    fig.supylabel("$\\hat\\tau_i$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, markerscale=5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    guardar(fig, "acic_tau")


if __name__ == "__main__":
    if sys.argv[1] == "resultados":
        experimento_positividad()
        experimento_acic()
    elif sys.argv[1] == "figuras":
        figura_positividad_datos()
        figura_positividad_metricas()
        figura_acic_propension()
        figura_acic_metricas()
        figura_acic_tau()
