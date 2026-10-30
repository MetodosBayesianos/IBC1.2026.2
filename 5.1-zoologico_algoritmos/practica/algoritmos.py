"""Algoritmos del zoologico con una interfaz comun: reciben X, t, y y opcionalmente X_grilla."""
import copy
import numpy as np
import torch
from econml.dml import CausalForestDML, LinearDML
from econml.dr import ForestDRLearner, LinearDRLearner
from econml.inference import BootstrapInference
from econml.metalearners import SLearner, TLearner
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from stochtree import BARTModel, BCFModel
from torch import nn

CLAVES = ["ate", "ate_inf", "ate_sup", "tau", "tau_inf", "tau_sup",
          "tau_grilla", "tau_grilla_inf", "tau_grilla_sup",
          "mu0_grilla", "mu0_grilla_inf", "mu0_grilla_sup",
          "mu1_grilla", "mu1_grilla_inf", "mu1_grilla_sup", "e_grilla"]


def armar_resultado(ate, ate_inf, ate_sup, **campos):
    """Diccionario con todas las claves de la interfaz; las que faltan valen None."""
    return {clave: None for clave in CLAVES} | {"ate": ate, "ate_inf": ate_inf, "ate_sup": ate_sup} | campos


def logistica():
    """Regresion logistica con penalizacion elegida por validacion cruzada sobre la log-verosimilitud."""
    # sin penalizar, con muchas covariables, algun fold del cross-fitting casi separa los datos y AIPW explota
    # l1_ratios y use_legacy_attributes fijan el comportamiento actual de scikit-learn 1.9 y evitan sus avisos de cambio;
    # EconML lee los atributos en el formato viejo
    return LogisticRegressionCV(Cs=5, scoring="neg_log_loss", max_iter=1000, l1_ratios=(0,), use_legacy_attributes=True)


def propension(X, t, X_grilla=None):
    """Propensity score estimado con regresion logistica, en X y en la grilla."""
    modelo = logistica().fit(X, t)
    e_grilla = None if X_grilla is None else modelo.predict_proba(X_grilla)[:, 1]
    return modelo.predict_proba(X)[:, 1], e_grilla


def intervalo_bootstrap(estimar_ate, X, t, y, n_boot, semilla):
    """Intervalo del 95% por percentiles del ATE sobre remuestreos de individuos."""
    rng = np.random.default_rng(semilla)
    ates = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        ates.append(estimar_ate(X[i], t[i], y[i]))
    return np.percentile(ates, [2.5, 97.5])


def ingenua(X, t, y, X_grilla=None):
    """Diferencia de medias entre tratados y controles, con intervalo normal."""
    ate = y[t == 1].mean() - y[t == 0].mean()
    error = np.sqrt(y[t == 1].var(ddof=1) / (t == 1).sum() + y[t == 0].var(ddof=1) / (t == 0).sum())
    return armar_resultado(ate, ate - 1.96 * error, ate + 1.96 * error)


def tau_matching(X, t, y):
    """Efecto de cada individuo imputando su contrafactual con el vecino mas cercano del otro grupo."""
    Z = StandardScaler().fit_transform(X)
    contrafactual = np.empty(len(y))
    for grupo in (0, 1):
        vecinos = NearestNeighbors(n_neighbors=1).fit(Z[t != grupo])
        _, j = vecinos.kneighbors(Z[t == grupo])
        contrafactual[t == grupo] = y[t != grupo][j[:, 0]]
    return np.where(t == 1, y - contrafactual, contrafactual - y)


def matching(X, t, y, X_grilla=None, n_boot=200, semilla=0):
    """Matching por vecino mas cercano con reemplazo sobre covariables estandarizadas."""
    tau = tau_matching(X, t, y)
    inf, sup = intervalo_bootstrap(lambda X, t, y: tau_matching(X, t, y).mean(), X, t, y, n_boot, semilla)
    return armar_resultado(tau.mean(), inf, sup, tau=tau)


def matching_propension(X, t, y, X_grilla=None, n_boot=200, semilla=0):
    """Matching por vecino mas cercano en el logit del propensity score."""
    # e se estima una sola vez: reestimarlo en cada remuestreo multiplica el tiempo, y para el ATE
    # no propagar su incertidumbre deja un intervalo conservador (Abadie e Imbens 2016)
    e, _ = propension(X, t)
    return matching(np.log(e / (1 - e))[:, None], t, y, n_boot=n_boot, semilla=semilla)


def ate_ipw(X, t, y):
    """ATE con pesos de Horvitz-Thompson, sin normalizar ni recortar."""
    e, _ = propension(X, t)
    return np.mean(t * y / e - (1 - t) * y / (1 - e))


def ipw(X, t, y, X_grilla=None, n_boot=200, semilla=0):
    """Inverse probability weighting con intervalo bootstrap."""
    inf, sup = intervalo_bootstrap(ate_ipw, X, t, y, n_boot, semilla)
    return armar_resultado(ate_ipw(X, t, y), inf, sup)


def efectos_econml(modelo, X, X_grilla):
    """ATE, tau en X y tau en la grilla, con sus intervalos, de un estimador de EconML ya ajustado."""
    campos = dict(zip(["tau_inf", "tau_sup"], modelo.effect_interval(X)), tau=modelo.effect(X))
    if X_grilla is not None:
        inf, sup = modelo.effect_interval(X_grilla)
        campos |= dict(tau_grilla=modelo.effect(X_grilla), tau_grilla_inf=inf, tau_grilla_sup=sup)
    return armar_resultado(modelo.ate(X), *modelo.ate_interval(X), **campos)


def s_learner(X, t, y, X_grilla=None, n_boot=50, semilla=0):
    """Un solo modelo de y sobre (X, t); el efecto es la diferencia de sus predicciones."""
    modelo = SLearner(overall_model=GradientBoostingRegressor(random_state=semilla))
    modelo.fit(y, t, X=X, inference=BootstrapInference(n_bootstrap_samples=n_boot))
    r = efectos_econml(modelo, X, X_grilla)
    if X_grilla is not None:
        # EconML codifica el tratamiento como las columnas (1 - t, t) al lado de X
        unos, ceros = np.ones((len(X_grilla), 1)), np.zeros((len(X_grilla), 1))
        r["mu0_grilla"] = modelo.overall_model.predict(np.hstack([X_grilla, unos, ceros]))
        r["mu1_grilla"] = modelo.overall_model.predict(np.hstack([X_grilla, ceros, unos]))
    return r


def t_learner(X, t, y, X_grilla=None, n_boot=50, semilla=0):
    """Un modelo de y por grupo; el efecto es la diferencia de sus predicciones."""
    modelo = TLearner(models=GradientBoostingRegressor(random_state=semilla))
    modelo.fit(y, t, X=X, inference=BootstrapInference(n_bootstrap_samples=n_boot))
    r = efectos_econml(modelo, X, X_grilla)
    if X_grilla is not None:
        r["mu0_grilla"] = modelo.models[0].predict(X_grilla)
        r["mu1_grilla"] = modelo.models[1].predict(X_grilla)
    return r


def doble_robusto(X, t, y, X_grilla=None, modelo_resultado=None, modelo_propension=None, semilla=0):
    """ATE por AIPW con cross-fitting y tau con un bosque sobre los pseudo-resultados."""
    modelos = dict(model_regression=modelo_resultado or GradientBoostingRegressor(random_state=semilla),
                   model_propensity=modelo_propension or logistica(), random_state=semilla)
    aipw = LinearDRLearner(**modelos).fit(y, t, X=None, W=X)
    bosque = ForestDRLearner(**modelos).fit(y, t, X=X)
    inf, sup = aipw.ate_interval()
    return efectos_econml(bosque, X, X_grilla) | dict(ate=aipw.ate(), ate_inf=inf, ate_sup=sup)


def dml(X, t, y, X_grilla=None, semilla=0):
    """DML parcialmente lineal: pendiente de los residuos de y contra los de t."""
    modelo = LinearDML(model_y=GradientBoostingRegressor(random_state=semilla), model_t=logistica(),
                       discrete_treatment=True, random_state=semilla).fit(y, t, X=None, W=X)
    return armar_resultado(modelo.ate(), *modelo.ate_interval())


def causal_forest(X, t, y, X_grilla=None, semilla=0):
    """Causal Forest con honest splitting sobre los residuos de DML."""
    modelo = CausalForestDML(model_y=GradientBoostingRegressor(random_state=semilla), model_t=logistica(),
                             discrete_treatment=True, random_state=semilla).fit(y, t, X=X)
    return efectos_econml(modelo, X, X_grilla)


class Red(nn.Module):
    """Representacion compartida de las covariables, una cabeza de resultado por grupo y una de propensity."""

    def __init__(self, n_covariables):
        super().__init__()
        self.representacion = nn.Sequential(nn.Linear(n_covariables, 200), nn.ELU(), nn.Linear(200, 200), nn.ELU(),
                                            nn.Linear(200, 200), nn.ELU())
        self.cabezas = nn.ModuleList([nn.Sequential(nn.Linear(200, 100), nn.ELU(), nn.Linear(100, 100), nn.ELU(),
                                                    nn.Linear(100, 1)) for _ in range(2)])
        self.propension = nn.Linear(200, 1)
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        phi = self.representacion(X)
        return self.cabezas[0](phi)[:, 0], self.cabezas[1](phi)[:, 0], torch.sigmoid(self.propension(phi)[:, 0])


def perdida_red(red, X, t, y, dragon):
    """Perdida de TARNet, o de Dragonnet con la cabeza de propensity y la regularizacion dirigida, sumada sobre individuos."""
    mu0, mu1, e = red(X)
    mu = torch.where(t == 1, mu1, mu0)
    perdida = ((y - mu) ** 2).sum()
    if dragon:
        perdida = perdida + nn.functional.binary_cross_entropy(e, t, reduction="sum")
        e = e.clamp(0.01, 0.99)
        # en el optimo de epsilon, el promedio de los residuos pesados se anula: la correccion de AIPW
        perdida = perdida + ((y - mu - red.epsilon * (t / e - (1 - t) / (1 - e))) ** 2).sum()
    return perdida


def entrenar_red(X, t, y, dragon, semilla):
    """Red entrenada sobre un remuestreo bootstrap, con early stopping sobre los individuos que quedaron afuera."""
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    bolsa = rng.integers(0, len(y), len(y))
    afuera = torch.from_numpy(np.setdiff1d(np.arange(len(y)), bolsa))
    red = Red(X.shape[1])
    optimizador = torch.optim.Adam(red.parameters(), lr=0.001)
    pesos_cabezas = [p for nombre, p in red.cabezas.named_parameters() if "weight" in nombre]
    mejor, sin_mejorar = np.inf, 0
    for _ in range(300):
        for lote in np.array_split(rng.permutation(bolsa), int(np.ceil(len(bolsa) / 64))):
            lote = torch.from_numpy(lote)
            optimizador.zero_grad()
            l2 = 0.01 * sum((p ** 2).sum() for p in pesos_cabezas)
            (perdida_red(red, X[lote], t[lote], y[lote], dragon) + l2).backward()
            optimizador.step()
        with torch.no_grad():
            validacion = perdida_red(red, X[afuera], t[afuera], y[afuera], dragon).item()
        if validacion < mejor:
            mejor, sin_mejorar, estado = validacion, 0, copy.deepcopy(red.state_dict())
        else:
            sin_mejorar += 1
            if sin_mejorar == 20:
                break
    red.load_state_dict(estado)
    return red.eval()


def resumen_ensamble(muestras):
    """Promedio de las redes y promedio +/- 1.96 desvios entre redes."""
    media, desvio = muestras.mean(axis=0), muestras.std(axis=0, ddof=1)
    return media, media - 1.96 * desvio, media + 1.96 * desvio


def ensamble_red(X, t, y, X_grilla, dragon, semilla, n_redes):
    """Ensamble de redes, cada una sobre un remuestreo bootstrap, con covariables e y estandarizadas."""
    escala = StandardScaler().fit(X)
    media, desvio = y.mean(), y.std()
    datos = [torch.tensor(a, dtype=torch.float32) for a in (escala.transform(X), t, (y - media) / desvio)]
    # semilla * n_redes + k: con semilla + k, las replicas r y r + 1 de IHDP compartirian 9 de 10 redes
    redes = [entrenar_red(*datos, dragon, semilla * n_redes + k) for k in range(n_redes)]

    def predecir(A):
        """mu0, mu1 y e de cada red (una fila por red) en la escala original de y."""
        with torch.no_grad():
            salidas = [red(torch.tensor(escala.transform(A), dtype=torch.float32)) for red in redes]
        mu0, mu1, e = (np.array([s[i].numpy() for s in salidas]) for i in range(3))
        return mu0 * desvio + media, mu1 * desvio + media, e

    mu0, mu1, e = predecir(X)
    tau = mu1 - mu0
    ates = tau.mean(axis=1)
    if dragon:
        # el ATE usa las curvas corregidas por la regularizacion dirigida; tau y las curvas no, porque
        # la correccion por individuo explota donde e esta cerca de 0 o 1
        e = np.clip(e, 0.01, 0.99)
        epsilon = np.array([[red.epsilon.item()] for red in redes]) * desvio
        ates = (tau + epsilon * (1 / e + 1 / (1 - e))).mean(axis=1)
    campos = dict(zip(["tau", "tau_inf", "tau_sup"], resumen_ensamble(tau)))
    if X_grilla is not None:
        mu0_grilla, mu1_grilla, e_grilla = predecir(X_grilla)
        campos |= dict(zip(["tau_grilla", "tau_grilla_inf", "tau_grilla_sup"], resumen_ensamble(mu1_grilla - mu0_grilla)))
        campos |= dict(zip(["mu0_grilla", "mu0_grilla_inf", "mu0_grilla_sup"], resumen_ensamble(mu0_grilla)))
        campos |= dict(zip(["mu1_grilla", "mu1_grilla_inf", "mu1_grilla_sup"], resumen_ensamble(mu1_grilla)))
        if dragon:
            campos["e_grilla"] = e_grilla.mean(axis=0)
    return armar_resultado(*resumen_ensamble(ates), **campos)


def tarnet(X, t, y, X_grilla=None, semilla=0, n_redes=10):
    """TARNet: representacion compartida y una cabeza de resultado por grupo."""
    return ensamble_red(X, t, y, X_grilla, False, semilla, n_redes)


def dragonnet(X, t, y, X_grilla=None, semilla=0, n_redes=10):
    """Dragonnet: TARNet con cabeza de propensity y regularizacion dirigida."""
    return ensamble_red(X, t, y, X_grilla, True, semilla, n_redes)


def resumen(muestras):
    """Media e intervalo del 95% de cada fila de muestras de la posterior."""
    return muestras.mean(axis=1), *np.percentile(muestras, [2.5, 97.5], axis=1)


def resultado_posterior(tau_muestras, tau_grilla_muestras=None, **campos):
    """ATE, tau y tau en la grilla a partir de muestras de la posterior de tau."""
    ate, ate_inf, ate_sup = resumen(tau_muestras.mean(axis=0)[None, :])
    tau, tau_inf, tau_sup = resumen(tau_muestras)
    if tau_grilla_muestras is not None:
        campos |= dict(zip(["tau_grilla", "tau_grilla_inf", "tau_grilla_sup"], resumen(tau_grilla_muestras)))
    return armar_resultado(ate[0], ate_inf[0], ate_sup[0], tau=tau, tau_inf=tau_inf, tau_sup=tau_sup, **campos)


def bart(X, t, y, X_grilla=None, semilla=0, n_burnin=500, n_mcmc=1000):
    """BART sobre (X, t); el efecto sale de predecir cada individuo con t = 1 y t = 0."""
    modelo = BARTModel()
    modelo.sample(np.c_[X, t], y, num_gfr=10, num_burnin=n_burnin, num_mcmc=n_mcmc,
                  general_params={"random_seed": semilla})

    def mu(X, tratamiento):
        return modelo.predict(np.c_[X, np.full(len(X), tratamiento)])["y_hat"]

    if X_grilla is None:
        return resultado_posterior(mu(X, 1) - mu(X, 0))
    mu0, mu1 = mu(X_grilla, 0), mu(X_grilla, 1)
    campos = dict(zip(["mu0_grilla", "mu0_grilla_inf", "mu0_grilla_sup"], resumen(mu0)))
    campos |= dict(zip(["mu1_grilla", "mu1_grilla_inf", "mu1_grilla_sup"], resumen(mu1)))
    return resultado_posterior(mu(X, 1) - mu(X, 0), mu1 - mu0, **campos)


def bcf(X, t, y, X_grilla=None, semilla=0, n_burnin=500, n_mcmc=1000):
    """Bayesian Causal Forest con el propensity de la logistica como covariable."""
    e, e_grilla = propension(X, t, X_grilla)
    modelo = BCFModel()
    modelo.sample(X, t, y, propensity_train=e, num_gfr=10, num_burnin=n_burnin, num_mcmc=n_mcmc,
                  general_params={"random_seed": semilla})
    tau_muestras = modelo.predict(X, t, propensity=e, terms="cate")
    if X_grilla is None:
        return resultado_posterior(tau_muestras)
    unos = np.ones(len(X_grilla))
    return resultado_posterior(tau_muestras, modelo.predict(X_grilla, unos, propensity=e_grilla, terms="cate"))


ALGORITMOS = {"ingenua": ingenua, "matching": matching, "ipw": ipw, "s_learner": s_learner,
              "t_learner": t_learner, "doble_robusto": doble_robusto, "dml": dml,
              "causal_forest": causal_forest, "tarnet": tarnet, "dragonnet": dragonnet, "bart": bart, "bcf": bcf}

NOMBRES = {"ingenua": "Diferencia ingenua", "matching": "Matching", "ps_matching": "PS matching", "ipw": "IPW",
           "s_learner": "S-Learner",
           "t_learner": "T-Learner", "doble_robusto": "Doubly robust", "dml": "DML",
           "causal_forest": "Causal Forest", "tarnet": "TARNet", "dragonnet": "Dragonnet", "bart": "BART", "bcf": "BCF"}
