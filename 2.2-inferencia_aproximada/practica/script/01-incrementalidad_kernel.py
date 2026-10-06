import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from matplotlib import pyplot as plt

# Datos: conversiones (n) y tamaño (N) de cada grupo, por mes
data = pd.read_csv("../datos/01-Incrementalidad.csv", encoding="utf-8-sig")
def serie(tipo):
    return data[data["Tipo"] == tipo]["online_schedules"].to_numpy()

meses = data["Mes"].unique()
T = len(meses)
n_C, N_C = serie("Eventos CG"), serie("Usuarios CG")
n_T, N_T = serie("Eventos Tratamiento"), serie("Usuarios Tratamiento")


# Modelo temporal, una cadena por grupo, cada una con su propio kappa:
#   log10(kappa) ~ U[0, 7]
#   p_0 ~ Beta(1, 20)
#   p_t | p_{t-1}, kappa ~ Beta(kappa p_{t-1}, kappa (1 - p_{t-1}))
#   n_t | p_t ~ Binomial(n_t | p_t, N_t)

def cadena(nombre, n, N):
    log10_kappa = pm.Uniform(f"log10_kappa_{nombre}", 0, 7)
    kappa = 10 ** log10_kappa
    p = [pm.Beta(f"{nombre}_0", alpha=1, beta=20)]
    for t in range(1, T):
        p.append(pm.Beta(f"{nombre}_{t}", alpha=kappa * p[t - 1], beta=kappa * (1 - p[t - 1])))
    p = pm.Deterministic(nombre, pt.stack(p))
    pm.Binomial(f"n_{nombre}", n=N, p=p, observed=n)

with pm.Model():
    cadena("p", n_C, N_C)  # Desempeño en control
    cadena("q", n_T, N_T)  # Desempeño en tratamiento
    traza = pm.sample(2000, tune=2000, chains=4, target_accept=0.99)


# Diagnósticos: divergencias, r_hat (debe ser < 1.01) y tamaño efectivo de muestra (ESS)
resumen = az.summary(traza)
print(f"\n{traza.sample_stats['diverging'].values.sum()} divergencias, "
      f"r_hat máximo {resumen['r_hat'].max():.3f}, ESS mínimo {resumen['ess_bulk'].min():.0f}")
print(resumen.loc[["log10_kappa_p", "log10_kappa_q"], ["mean", "sd", "ess_bulk", "r_hat"]])


# Resultados
# P(q_t > p_t) se calcula con las muestras conjuntas del posterior.
# Para comparar, el modelo base con cada mes por separado: Beta(1 + n, 20 + N - n).
p = traza.posterior["p"].values.reshape(-1, T)
q = traza.posterior["q"].values.reshape(-1, T)
p_kernel = (q > p).mean(axis=0)
p_mes = (np.random.beta(1 + n_T, 20 + N_T - n_T, (100000, T)) >
         np.random.beta(1 + n_C, 20 + N_C - n_C, (100000, T))).mean(axis=0)

print("\nMes     P(q>p) kernel  P(q>p) mes por separado")
for t, mes in enumerate(meses):
    print(f"{mes}  {p_kernel[t]:13.3f}  {p_mes[t]:23.3f}")


# Gráficos
x = np.arange(T)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11))
for c, (nombre, muestras, n, N) in enumerate([("Control", p, n_C, N_C), ("Tratamiento", q, n_T, N_T)]):
    ax1.plot(x, muestras.mean(axis=0), color=f"C{c}", lw=2, label=f"{nombre} (media posterior)")
    ax1.fill_between(x, *np.percentile(muestras, [2.5, 97.5], axis=0), color=f"C{c}", alpha=0.2)
    ax1.plot(x, n / N, "o", color=f"C{c}", label=f"{nombre} (frecuencia observada)")
ax1.set_xticks(x, meses)
ax1.set_ylabel("Desempeño")
ax1.legend(fontsize=8)
ax2.plot(x, p_kernel, "k-o", label="Modelo temporal")
ax2.plot(x, p_mes, "o--", color="gray", label="Mes por separado")
ax2.axhline(0.95, color="k", ls=":", lw=1)
ax2.set_xticks(x, meses)
ax2.set_ylim(0, 1.02)
ax2.set_ylabel("P(q > p | datos)")
ax2.legend(fontsize=8)
for c, nombre in enumerate(["p", "q"]):
    ax3.hist(traza.posterior[f"log10_kappa_{nombre}"].values.ravel(), bins=50, range=(0, 7),
             density=True, color=f"C{c}", alpha=0.5, label=nombre)
ax3.set_xlabel("log10(kappa)")
ax3.set_ylabel("P(log10 kappa | datos)")
ax3.legend(fontsize=8)
plt.tight_layout()
plt.savefig("incrementalidad_kernel.pdf")
