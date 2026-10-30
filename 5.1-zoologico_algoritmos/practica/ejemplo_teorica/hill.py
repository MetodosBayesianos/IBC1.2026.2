"""Simulacion de Hill (2011) con Q como fork, corregida en la tesis de Constanza De Galvagni."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from algoritmos import ALGORITMOS, NOMBRES, doble_robusto, logistica, propension

plt.rcParams.update({"font.size": 16})
cmap = plt.get_cmap("tab10")

SEMILLA_HILL = 128  # primera semilla con pesos de IPW mayores a 100 y a lo sumo 30 puntos fuera del solapamiento


def mu0(q):
    return 72 + 3 * np.sqrt(np.abs(q))


def mu1(q):
    return 90 + np.exp(0.06 * q)


def tau(q):
    return mu1(q) - mu0(q)


def e_verdadero(q):
    return 1 / (1 + np.exp(-(q / 5 - 6)))


def densidad_q(q):
    return 0.5 * norm.pdf(q, 20, 10) + 0.5 * norm.pdf(q, 40, 10)


def simular_hill(n=120, semilla=SEMILLA_HILL):
    """Q de una mezcla de normales, T con P(t|q) = sigmoide(q/5 - 6) e Y normal alrededor de mu_t(q)."""
    rng = np.random.default_rng(semilla)
    q = rng.normal(np.where(rng.random(n) < 0.5, 20, 40), 10)
    t = rng.binomial(1, e_verdadero(q))
    y = rng.normal(np.where(t == 1, mu1(q), mu0(q)), 1)
    return q, t, y


def ate_mezcla():
    """ATE poblacional, E[tau(Q)] bajo la mezcla."""
    return quad(lambda q: tau(q) * densidad_q(q), -60, 120)[0]


def estimando_dml(q):
    """Lo que estima el DML parcialmente lineal: tau promediado con pesos e(1 - e)."""
    pesos = e_verdadero(q) * (1 - e_verdadero(q))
    return np.sum(pesos * tau(q)) / np.sum(pesos)


def zona_solapamiento(q, t):
    """Intervalo de q donde hay datos de los dos grupos."""
    return max(q[t == 0].min(), q[t == 1].min()), min(q[t == 0].max(), q[t == 1].max())


QLIM = (-13, 65)
GRILLA = np.linspace(*QLIM, 200)
YLIM = (70, 135)
TAULIM = (-5, 45)
TABLEROLIM = (0, 30)
COLOR_CONTROL, COLOR_TRATADO, COLOR_ESTIMADO = cmap(0), cmap(1), cmap(2)


def ejes_q(ylabel, ylim, q, t, sombrear=True):
    """Figura con eje Q fijo y la zona sin solapamiento sombreada."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(QLIM)
    ax.set_ylim(ylim)
    ax.set_xlabel("$Q$")
    ax.set_ylabel(ylabel)
    if sombrear:
        inferior, superior = zona_solapamiento(q, t)
        ax.axvspan(QLIM[0], inferior, color="gray", alpha=0.15, lw=0)
        ax.axvspan(superior, QLIM[1], color="gray", alpha=0.15, lw=0)
    return fig, ax


def puntos(ax, q, t, y, tamanos=40):
    ax.scatter(q[t == 0], y[t == 0], s=tamanos if np.isscalar(tamanos) else tamanos[t == 0],
               color=COLOR_CONTROL, alpha=0.6, label="Controles")
    ax.scatter(q[t == 1], y[t == 1], s=tamanos if np.isscalar(tamanos) else tamanos[t == 1],
               color=COLOR_TRATADO, alpha=0.6, marker="^", label="Tratados")


def curvas_verdaderas(ax):
    ax.plot(GRILLA, mu0(GRILLA), color=COLOR_CONTROL, lw=2, label="$\\mu_0$ verdadera")
    ax.plot(GRILLA, mu1(GRILLA), color=COLOR_TRATADO, lw=2, label="$\\mu_1$ verdadera")


def tau_verdadero(ax):
    ax.plot(GRILLA, tau(GRILLA), color="black", lw=2, label="$\\tau(q)$ verdadero")


def estimacion_tau(ax, r, etiqueta="Estimado"):
    """Tau estimado en la grilla con su banda, o el ATE como recta horizontal si no hay tau."""
    if r["tau_grilla"] is None:
        ax.axhline(r["ate"], color=COLOR_ESTIMADO, lw=2, label=etiqueta + " (solo ATE)")
        ax.axhspan(r["ate_inf"], r["ate_sup"], color=COLOR_ESTIMADO, alpha=0.2, lw=0)
        return
    ax.plot(GRILLA, r["tau_grilla"], color=COLOR_ESTIMADO, lw=2, label=etiqueta)
    if r["tau_grilla_inf"] is not None:
        ax.fill_between(GRILLA, r["tau_grilla_inf"], r["tau_grilla_sup"], color=COLOR_ESTIMADO, alpha=0.2, lw=0)


def curvas_estimadas(ax, r):
    for grupo, color in (("mu0", COLOR_CONTROL), ("mu1", COLOR_TRATADO)):
        ax.plot(GRILLA, r[grupo + "_grilla"], color=color, lw=2, ls="--", label=f"$\\hat\\mu_{grupo[-1]}$")
        if r[grupo + "_grilla_inf"] is not None:
            ax.fill_between(GRILLA, r[grupo + "_grilla_inf"], r[grupo + "_grilla_sup"], color=color, alpha=0.2, lw=0)


def guardar(paginas, nombre, tamano_leyenda=11, lugar_leyenda="upper left"):
    with PdfPages(nombre) as pdf:
        for fig in paginas:
            if lugar_leyenda is not None:
                fig.axes[0].legend(loc=lugar_leyenda, fontsize=tamano_leyenda)
            pdf.savefig(fig)
            plt.close(fig)


def figura_datos(q, t, y):
    paginas = []
    for paso in range(3):
        fig, ax = ejes_q("$Y$", YLIM, q, t, sombrear=paso == 2)
        puntos(ax, q, t, y)
        if paso >= 1:
            curvas_verdaderas(ax)
        paginas.append(fig)
    guardar(paginas, "hill_datos.pdf")


def figura_tau(q, t, r, nombre, mecanismos=()):
    """PDF con las paginas de mecanismo (opcionales) y una de tau estimado sobre el verdadero."""
    fig, ax = ejes_q("$\\tau$", TAULIM, q, t)
    tau_verdadero(ax)
    estimacion_tau(ax, r)
    guardar([*mecanismos, fig], nombre)


def mecanismo_matching(q, t, y):
    fig, ax = ejes_q("$Y$", YLIM, q, t)
    vecino = {}
    for grupo in (0, 1):
        otros = np.flatnonzero(t != grupo)
        for i in np.flatnonzero(t == grupo):
            vecino[i] = otros[np.argmin(np.abs(q[otros] - q[i]))]
    for i, j in vecino.items():
        ax.plot([q[i], q[j]], [y[i], y[j]], color="gray", lw=0.8)
    puntos(ax, q, t, y)
    return fig


def figura_matching(q, t, y, r):
    fig, ax = ejes_q("$\\tau$", TAULIM, q, t)
    tau_verdadero(ax)
    ax.scatter(q, r["tau"], color=COLOR_ESTIMADO, s=25, label="$\\hat\\tau_i$")
    guardar([mecanismo_matching(q, t, y), fig], "hill_matching.pdf")


def figura_e(q, t, e_grilla, etiqueta):
    """Propensity estimado en la grilla sobre el verdadero, con los tratamientos observados en 0 y 1."""
    fig, ax = ejes_q("$P(T=1|q)$", (-0.05, 1.05), q, t)
    ax.plot(GRILLA, e_verdadero(GRILLA), color="black", lw=2, label="$e(q)$ verdadero")
    ax.plot(GRILLA, e_grilla, color=COLOR_ESTIMADO, lw=2, ls="--", label=etiqueta)
    ax.scatter(q[t == 0], np.zeros((t == 0).sum()), color=COLOR_CONTROL, alpha=0.6, label="Controles")
    ax.scatter(q[t == 1], np.ones((t == 1).sum()), color=COLOR_TRATADO, alpha=0.6, marker="^", label="Tratados")
    return fig


def figura_propension(q, t):
    _, e_grilla = propension(q[:, None], t, GRILLA[:, None])
    guardar([figura_e(q, t, e_grilla, "$\\hat e(q)$")], "hill_propension.pdf")


def figura_ps_matching(q, t):
    """Solapamiento de e estimado por grupo, y balance de Q antes y despues de matchear a los tratados."""
    e, _ = propension(q[:, None], t)
    logit = np.log(e / (1 - e))
    controles = np.flatnonzero(t == 0)
    elegidos = [controles[np.argmin(np.abs(logit[controles] - logit[i]))] for i in np.flatnonzero(t == 1)]
    fig_e, ax = plt.subplots(figsize=(8, 5))
    bordes = np.linspace(0, 1, 21)
    ax.hist(e[t == 0], bins=bordes, color=COLOR_CONTROL, alpha=0.6, label="Controles")
    ax.hist(e[t == 1], bins=bordes, color=COLOR_TRATADO, alpha=0.6, label="Tratados")
    ax.set_xlabel("$\\hat e(q)$")
    ax.set_ylabel("Individuos")
    paginas = [fig_e]
    bordes = np.linspace(*QLIM, 27)
    for q_controles, etiqueta in ((q[controles], "Controles"), (q[elegidos], "Controles elegidos")):
        fig, ax = ejes_q("Densidad", (0, 0.1), q, t)
        ax.hist(q_controles, bins=bordes, density=True, color=COLOR_CONTROL, alpha=0.6, label=etiqueta)
        ax.hist(q[t == 1], bins=bordes, density=True, color=COLOR_TRATADO, alpha=0.6, label="Tratados")
        paginas.append(fig)
    guardar(paginas, "hill_ps_matching.pdf", lugar_leyenda="upper right")


def mecanismo_ipw(q, t, y):
    e, _ = propension(q[:, None], t)
    pesos = np.where(t == 1, 1 / e, 1 / (1 - e))
    fig, ax = ejes_q("$Y$", YLIM, q, t)
    puntos(ax, q, t, y, tamanos=10 * pesos)
    return fig


def figura_curvas(q, t, y, r, nombre, previas=()):
    """Paginas previas (opcionales), pagina con las curvas estimadas de cada grupo y pagina con tau estimado."""
    fig, ax = ejes_q("$Y$", YLIM, q, t)
    puntos(ax, q, t, y)
    curvas_verdaderas(ax)
    curvas_estimadas(ax, r)
    figura_tau(q, t, r, nombre, mecanismos=[*previas, fig])


def figura_doble_robusto(q, t, y):
    variantes = [doble_robusto(q[:, None], t, y, GRILLA[:, None]),
                 doble_robusto(q[:, None], t, y, GRILLA[:, None], modelo_resultado=LinearRegression()),
                 doble_robusto(q[:, None], t, y, GRILLA[:, None], modelo_propension=DummyClassifier(strategy="prior"))]
    etiquetas = ["Doubly robust", "Resultado lineal", "Propensity constante"]
    paginas = []
    for r, etiqueta in zip(variantes, etiquetas):
        fig, ax = ejes_q("$\\tau$", TAULIM, q, t)
        tau_verdadero(ax)
        estimacion_tau(ax, r, etiqueta)
        ax.axhline(tau(q).mean(), color="black", lw=1, ls=":", label="ATE verdadero")
        ax.axhline(r["ate"], color=COLOR_ESTIMADO, lw=1, ls=":", label="ATE (AIPW)")
        paginas.append(fig)
    guardar(paginas, "hill_doble_robusto.pdf")


def mecanismo_dml(q, t, y):
    m = cross_val_predict(GradientBoostingRegressor(random_state=0), q[:, None], y, cv=5)
    e = cross_val_predict(logistica(), q[:, None], t, cv=5, method="predict_proba")[:, 1]
    residuo_t, residuo_y = t - e, y - m
    pendiente = np.sum(residuo_t * residuo_y) / np.sum(residuo_t ** 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(residuo_t, residuo_y, color="gray", alpha=0.6, label="Individuos")
    extremos = np.array([residuo_t.min(), residuo_t.max()])
    ax.plot(extremos, pendiente * extremos, color=COLOR_ESTIMADO, lw=2, label=f"Pendiente {pendiente:.1f}")
    ax.set_xlabel("$T - \\hat e(q)$")
    ax.set_ylabel("$Y - \\hat m(q)$")
    return fig


def figura_dml(q, t, y, r):
    fig, ax = ejes_q("$\\tau$", TAULIM, q, t)
    tau_verdadero(ax)
    estimacion_tau(ax, r, "DML")
    ax.axhline(estimando_dml(q), color="black", lw=1, ls="--", label="Estimando de DML")
    guardar([mecanismo_dml(q, t, y), fig], "hill_dml.pdf")


def figura_tablero(q, resultados):
    """Pagina k con los primeros k algoritmos: ATE estimado e intervalo contra el verdadero."""
    nombres = list(resultados)
    paginas = []
    for k in range(1, len(nombres) + 1):
        # letra grande: en la diapositiva el tablero ocupa un tercio del ancho
        plt.rcParams.update({"font.size": 24})
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(TABLEROLIM)
        ax.set_ylim(len(nombres) - 0.5, -0.5)
        # con 12 filas los nombres de 24 puntos se pisan
        ax.set_yticks(range(len(nombres)), [NOMBRES[n] if i < k else "" for i, n in enumerate(nombres)], fontsize=18)
        ax.axvline(tau(q).mean(), color="black", lw=2)
        ax.text(tau(q).mean() + 0.5, -0.6, "Verdadero", fontsize=18)
        if k > nombres.index("dml"):
            ax.plot(estimando_dml(q), nombres.index("dml"), marker="|", ms=20, color="black")
        for i, n in enumerate(nombres[:k]):
            r = resultados[n]
            ax.plot([r["ate_inf"], r["ate_sup"]], [i, i], color=COLOR_ESTIMADO, lw=2)
            if TABLEROLIM[0] <= r["ate"] <= TABLEROLIM[1]:
                ax.plot(r["ate"], i, "o", color=COLOR_ESTIMADO, ms=8)
            else:
                # fuera de escala: flecha en el borde con el valor
                borde = TABLEROLIM[0] if r["ate"] < TABLEROLIM[0] else TABLEROLIM[1]
                ax.plot(borde, i, "<" if r["ate"] < TABLEROLIM[0] else ">", color=COLOR_ESTIMADO, ms=12, clip_on=False)
                ax.annotate(f"{r['ate']:.0f}", (borde, i), xytext=(12 if borde == TABLEROLIM[0] else -40, -12),
                            textcoords="offset points", fontsize=18)
        ax.set_xlabel("ATE")
        fig.tight_layout()
        paginas.append(fig)
    plt.rcParams.update({"font.size": 16})
    guardar(paginas, "hill_tablero.pdf", lugar_leyenda=None)


def figura_comparacion(q, t, resultados):
    # con las redes son mas de 10 algoritmos: a tab10 se le suman dos colores oscuros que no se confunden con los suyos
    colores = plt.get_cmap("tab10").colors + ("navy", "darkred")
    fig, ax = ejes_q("$\\tau$", TAULIM, q, t)
    for i, (n, r) in enumerate(resultados.items()):
        if n in ("ingenua", "matching"):
            continue
        if r["tau_grilla"] is None:
            ax.axhline(r["ate"], color=colores[i], lw=1.5, ls="--", label=NOMBRES[n])
        elif n != "matching":
            ax.plot(GRILLA, r["tau_grilla"], color=colores[i], lw=1.5, label=NOMBRES[n])
    ax.scatter(q, resultados["matching"]["tau"], color=colores[1], s=12, label="Matching")  # matching no da tau en la grilla
    ax.plot(GRILLA, tau(GRILLA), color="black", lw=3, label="$\\tau(q)$ verdadero")
    guardar([fig], "hill_comparacion.pdf")


if __name__ == "__main__":
    q, t, y = simular_hill()
    X = q[:, None]
    resultados = {n: f(X, t, y, GRILLA[:, None]) for n, f in ALGORITMOS.items()}
    print({n: round(r["ate"], 2) for n, r in resultados.items()}, "verdadero", round(tau(q).mean(), 2))
    figura_datos(q, t, y)
    figura_matching(q, t, y, resultados["matching"])
    figura_propension(q, t)
    figura_ps_matching(q, t)
    figura_tau(q, t, resultados["ipw"], "hill_ipw.pdf", mecanismos=[mecanismo_ipw(q, t, y)])
    for n in ("s_learner", "t_learner", "tarnet", "bart"):
        figura_curvas(q, t, y, resultados[n], f"hill_{n}.pdf")
    figura_curvas(q, t, y, resultados["dragonnet"], "hill_dragonnet.pdf",
                  previas=[figura_e(q, t, resultados["dragonnet"]["e_grilla"], "Cabeza de propensity")])
    figura_doble_robusto(q, t, y)
    figura_dml(q, t, y, resultados["dml"])
    for n in ("causal_forest", "bcf"):
        figura_tau(q, t, resultados[n], f"hill_{n}.pdf")
    figura_tablero(q, resultados)
    figura_comparacion(q, t, resultados)
