import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import beta, betabinom

# Datos: conversiones (n) y tamaño (N) de cada grupo, por mes
data = pd.read_csv("../datos/01-Incrementalidad.csv", encoding="utf-8-sig")

def serie(tipo):
    return data[data["Tipo"] == tipo]["online_schedules"].to_numpy()

meses = data["Mes"].unique()
T = len(meses)

# Conversiones
n_C, N_C = serie("Eventos CG"), serie("Usuarios CG")
n_T, N_T = serie("Eventos Tratamiento"), serie("Usuarios Tratamiento")

# likelihood(p_t)
likelihood_cg = np.array([(int(n),int(N-n)) for n, N in zip(n_C,N_C)])
likelihood_tr = np.array([(int(n),int(N-n)) for n, N in zip(n_T,N_T)])
posterior_cg = np.zeros((T,2))
posterior_tr = np.zeros((T,2))

forwardPrior_cg = np.zeros((T,2)); forwardPrior_cg[t] = [1,20]
forwardPrior_tr = np.zeros((T,2))
backwardPrior_cg = np.zeros((T,2))
backwardPrior_tr = np.zeros((T,2))

_lambda = 0.5 # FALTA PORBAR UNA GRILLA DE VALORES Y GUARDARSE la log_evidencia

for t in range(T-1):#t=0
    posterior_cg[t] = likelihood_cg[t] + forwardPrior_cg[t] + backwardPrior_cg[t]
    forwardPosterior = likelihood_cg[t] + forwardPrior_cg[t]
    forwardPrior_cg[t+1] = _lambda * forwardPosterior
    #
    posterior_tr[t] = likelihood_tr[t] + forwardPrior_tr[t] + backwardPrior_tr[t]
    forwardPosterior = likelihood_tr[t] + forwardPrior_tr[t]
    forwardPrior_tr[t+1] = _lambda * forwardPosterior

for t in range(T-1,0,-1):
    posterior_cg[t] = likelihood_cg[t] + forwardPrior_cg[t] + backwardPrior_cg[t]
    backwardPosterior = likelihood_cg[t] + backwardPrior_cg[t]
    backwardPrior_cg[t-1] = _lambda * backwardPosterior
    #
    posterior_tr[t] = likelihood_tr[t] + forwardPrior_tr[t] + backwardPrior_tr[t]
    backwardPosterior = likelihood_tr[t] + backwardPrior_tr[t]
    backwardPrior_tr[t-1] = _lambda * backwardPosterior


log_evidencias_cg = [1] * T
log_evidencias_tr = [1] * T
for t in range(T):
    withinPrior_cg = forwardPrior_cg[t] + backwardPrior_cg[t]
    alpha_cg, beta_cg = withinPrior_cg
    log_evidencias_cg[t] = betabinom.logpmf(
        likelihood_cg[t][0],   # n
        sum(likelihood_cg[t]), # N
        alpha_cg, beta_cg) # Dist a priori
    #
    withinPrior_tr = forwardPrior_tr[t] + backwardPrior_tr[t]
    alpha_tr, beta_tr = withinPrior_tr
    log_evidencias_tr[t] = betabinom.logpmf(
        likelihood_tr[t][0],   # n
        sum(likelihood_tr[t]), # N
        alpha_tr, beta_tr) # Dist a priori

log_evidencia = sum(log_evidencias_cg) + sum(log_evidencias_tr)

lower_IC_cg = [beta.ppf(0.025, a_b[0], a_b[1]) for a_b in posterior_cg]
upper_IC_cg = [beta.ppf(1-0.025, a_b[0], a_b[1]) for a_b in posterior_cg]
middle_IC_cg = [beta.ppf(0.5, a_b[0], a_b[1]) for a_b in posterior_cg]

lower_IC_tr= [beta.ppf(0.025, a_b[0], a_b[1]) for a_b in posterior_tr]
upper_IC_tr = [beta.ppf(1-0.025, a_b[0], a_b[1]) for a_b in posterior_tr]
middle_IC_tr = [beta.ppf(0.5, a_b[0], a_b[1]) for a_b in posterior_tr]

plt.plot(np.arange(T),middle_IC_cg)
plt.plot(np.arange(T),middle_IC_tr)
plt.fill_between(np.arange(T), lower_IC_cg, upper_IC_cg, alpha=0.2, label='Control IC 95%')
plt.fill_between(np.arange(T), lower_IC_tr, upper_IC_tr, alpha=0.2, label='Treat IC 95%')
plt.show()







