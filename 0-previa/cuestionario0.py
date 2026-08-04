# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: global
#     language: python
#     name: python3
# ---

# %%
import math
import random
import inspect
import warnings

respuestas = {}

# %% [markdown]
# # 1 - Preguntas sobre fundamentos
#
# Este notebook contiene una lista de preguntas junto con una lista exhaustiva de respuestas mutuamente contradictorias. 
# A diferencia de los enunciados de tipo "multiple choise" en los que se pide seleccionar una única opción, aquí se pide que distribuyan creencias entre las diferentes opciones, asegurándose que el valor asignado sea positivo y la suma sea 1.
# La evaluación será el producto de las creencias asiganadas a las respuestas correctas. 
# En caso de que la respuesta sea una variable aleatoria, se considerará la predicción típica a largo plazo, es decir, su media geométrica.
# Notar que un único cero en la secuencia anula todo el producto.
# Por ello, en caso de duda, no conviene que concentren toda su creencia en una sola opción, sino distribuir algo de creencia en todas las opciones que consideran posibles.
# Notar también que conviene asignar más a la opción en la que más creen, porque distribuir creencias en partes iguales entre todas las opciones no es mucho mejor que el azar (baseline).

# %% [markdown]
# ### Moneda
#
# ¿Cuál será el resultado del lanzamiento de una moneda?
#
# 0. Anverso (Cara)
# 0. Reverso (Sello)
# 0. Canto (Borde)

# %%
respuestas["Moneda"] = [
0, # 0. Anverso (Cara)
0, # 1. Reverso (Sello)
0, # 2. Canto (Borde)
"""
Justifique
""",
]

# %% [markdown]
# ### 1.1 Cajas
#
# Hay tres cajas idénticas. Sabemos que detrás de una de ellas hay un regalo. El resto están vacías. ¿Dónde está el regalo?
#
# 0. Caja 0
# 1. Caja 1
# 2. Caja 2
# 3. Otro lugar

# %%
respuestas["Cajas"] = [
0, # 0. Caja 0
0, # 1. Caja 1
0, # 2. Caja 2
0, # 3. Otro lugar
"""
Justifique
""",
]

# %% [markdown]
# ### Mentir
#
# ¿Cuál de todas las opciones se considera una definición matemática del principio de no mentir?
#
# 0. Máxima incertidumbre (o entropía)
# 0. Mínima incertidumbre (o entropía)
# 0. Máxima incertidumbre (o entropía) dada la información disponible (restricciones)
# 0. Mínima incertidumbre (o entropía) dada la información disponible (restricciones)
# 0. Ninguna de las anteriores

# %%
respuestas["Mentir"] = [
0, # 0. Máxima incertidumbre (entropía)
0, # 1. Mínima incertidumbre (entropía)
0, # 2. Máxima incertidumbre (entropía) dada la información disponible (restricciones)
0, # 3. Mínima incertidumbre (entropía) dada la información disponible (restricciones)
0, # 4. Ninguna de las anteriores
"""
Justifique
""",
]

# %% [markdown]
# ### Universos
#
# En contextos de incertidumbre los posibles universos se bifurcan.
# Supongamos que hay tres cajas idénticas y detrás de una de ellas hay un regalo (el resto van a quedar vacías).
# Supongamos que nos permiten reservar una caja y luego, una persona nos muestra que en una de las otras cajas no está el regalo.
# En el contexto en el que reservamos la caja 1: ¿Cuál de todos los universos mutuamente contradictorios va a ocurrir?
# ¿El regalo está en la caja 1 y nos muestran la caja 1? ¿El regalo está en la caja 1 y nos muestran la caja 2?
# ... ¿El regalo está en la caja 3 y nos muestran la caja 2? ¿El regalo está en la caja 3 y nos muestran la caja 3?
#
# 0. Regalo = 1, Abren = 1
# 1. Regalo = 1, Abren = 2
# 2. Regalo = 1, Abren = 3
# 3. Regalo = 2, Abren = 1
# 4. Regalo = 2, Abren = 2
# 5. Regalo = 2, Abren = 3
# 6. Regalo = 3, Abren = 1
# 7. Regalo = 3, Abren = 2
# 8. Regalo = 3, Abren = 3
#

# %%
respuestas["Universos"] = [
0, # 0. Regalo = 1, Abren = 1
0, # 1. Regalo = 1, Abren = 2
0, # 2. Regalo = 1, Abren = 3
0, # 3. Regalo = 2, Abren = 1
0, # 4. Regalo = 2, Abren = 2
0, # 5. Regalo = 2, Abren = 3
0, # 6. Regalo = 3, Abren = 1
0, # 7. Regalo = 3, Abren = 2
0, # 8. Regalo = 3, Abren = 3
"""
Justifique
""",
]


# %% [markdown]
# ### Historia
#
# En los últimos siglos hubieron muchos avances científicos.
# En los últimos años, en particular, se han producido enormes avances en el área de la inteligencia artificial.
# ¿Cuándo ocurrió el primer uso conocido del actual sistema de razonamiento para contextos de incertidumbre?
#
# 0. Siglo 21
# 1. Siglo 20
# 2. Siglo 19
# 3. Siglo 18
# 4. Siglo 17
# 5. Antes

# %%
respuestas["Historia"] = [
0, # 0. Siglo 21
0, # 1. Siglo 20
0, # 2. Siglo 19
0, # 3. Siglo 18
0, # 4. Siglo 17
0, # 5. Antes
"""
Justifique
""",
]


# %% [markdown]
# ### Conjunta
#
# La distribución de creencias conjunta sobre varias variables se puede descomponer como el producto de la probabilidad marginal de cualquiera de las variables, multiplicado por la probabilidad condicional del resto de las variables dada la primera variable (usada en la marginal).
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Conjunta"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]


# %% [markdown]
# ### Independencia
#
# Si A es independiente de B sabemos que P(A|B) = P(B). ¿Pero es cierta la siguiente igualdad: P(A)P(B|A) = P(A)P(B)?
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Independencia"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]


# %% [markdown]
# ### Descomposiciones
#
# Sabemos que siempre existe alguna forma de descomponer la distribución conjunta como el producto de distribuciones condicionales unidimensionales.
# ¿Si hay N variables, cuántas descomposiciones en total existen?
#
# 0. 1
# 1. N - 1
# 2. N
# 3. N * (N - 1)
# 4. N * N
# 5. N! - 1
# 6. N!
# 7. N ^ N - 1
# 8. N ^ N

# %%
respuestas["Descomposiciones"] = [
0, # 0. 1
0, # 1. N - 1
0, # 2. N
0, # 3. N * (N - 1)
0, # 4. N * N
0, # 5. N! - 1
0, # 6. N!
0, # 7. N ^ N - 1
0, # 8. N ^ N
"""
Justifique
""",
]


# %% [markdown]
# ### Teorema de Bayes.
#
# El teorema de Bayes nos permite actualizar las creencias de las hipótesis internas a los modelos causales dado los datos.
# El denominador del teorema de Bayes es constante para las diferentes hipótesis.
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Teorema de Bayes"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]



# %% [markdown]
# ### Predicciones
#
# La predicción que una hipótesis H hace de un conjunto de datos, P(Datos = {d1, ..., d_n} | H), puede calcularse como el producto de las predicciones que la hipótesis hace de cada dato individual dado los datos ya vistos, P(d1|H)P(d2|d1,H)...
# Para que el cálculo sea correcto es importante que se respete el orden en el cual los datos fueron observados en los hechos, es decir, que no ocurra P(d2|H)P(d1|d2,H)..
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Predicciones"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]


# %% [markdown]
# ### Valor de verdad
#
# Si una hipótesis predice con 0 uno de los datos observados, la hipótesis se hace falsa.
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Valor de verdad"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]


# %% [markdown]
# ### Teorías causales
#
# Históricamente todas las ciencias con datos, desde la física hasta las ciencias sociales, explicaron el mundo a través de teorías causales.
# Los recientes avances en el área de aprendizaje automático e inteligencia artificial, sin embargo, se produjeron por el desarrollo de algoritmos altamente predictivos sin ninguna interpretación causal.
# ¿Qué relación hay entre los modelos causal y los complejos algoritmos de AI/ML?
#
# 0. El modelo causal que se corresponde con la realidad causal subyacente nunca puede ser mejor prediciendo que los complejos algoritmos de AI/ML.
# 1. El modelo causal que se corresponde con la realidad causal subyacente a veces puede ser mejor, y a veces peor, que los complejos algoritmos de AI/ML.
# 2. El modelo causal que se corresponde con la realidad causal subyacente nunca puede ser peor prediciendo que los complejos algoritmos de AI/ML.
# 3. No son comparables porque los modelos causales solo explican, no predicen.
# 4. Ninguna de las anteriores

# %%
respuestas["Teorías causales"] = [
0, # 0. El modelo causal que se corresponde con la realidad causal subyacente nunca puede ser mejor prediciendo que los complejos algoritmos de AI/ML.
0, # 1. El modelo causal que se corresponde con la realidad causal subyacente a veces puede ser mejor, y a veces peor, que los complejos algoritmos de AI/ML.
0, # 2. El modelo causal que se corresponde con la realidad causal subyacente nunca puede ser peor prediciendo que los complejos algoritmos de AI/ML.
0, # 3. No son comparables porque los modelos causales solo explican, no predicen.
0, # 4. Ninguna de las anteriores
"""
Justifique
""",
]


# %% [markdown]
# ### Predicción e información
#
# Cuanto mejor se predice más información (de Shannon) se obtiene.
#
# 0. Siempre
# 1. A veces
# 2. Nunca


# %%
respuestas["Predicción e información"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]


# %% [markdown]
# ### Modelos e información
#
# Al evaluar modelos causales, preferimos el que acumula más información (de Shannon).
#
# 0. Siempre
# 1. A veces
# 2. Nunca


# %%
respuestas["Modelos e información"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]

# %% [markdown]
# ### Evaluación de modelos
#
# Podemos identificar el modelo causal correcto si observamos suficientes datos.
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Evaluación de modelos"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]

# %% [markdown]
# ### Contrafactuales
#
# Cuando conocemos los mecasnimos causales probabilísticos de cada variable podemos usar la información factual para predecir cuál hubiera sido un resultado contrafactual.
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Contrafactuales"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]



# %% [markdown]
# ### Diversificación
#
# Una casa de apuestas nos paga 3 por Cara y 1.2 por Sello por el lanzamiento de moneda.
# La moneda es normal, con 0.5 de probabilidad de que salga Cara o Sello.
# Supongamos que nos ofrecen jugar 10000 veces, pero apostando absolutamente todos los recursos en cada paso temporal.
# Apostamos todo, nos devuelven actualizado y volvemos a apostar.
# ¿Qué proporción conviene apostar a Cara?
# Notar que el resto se asigna a Sello.
# Notar además que si apostamos todo a Cara y sale Sello perdemos todos los recursos y no podemos volver a jugar (solo nos pagan en el lado donde sale la moneda).
#
# 0. Recursos asignados a Cara: 0.0
# 1. Recursos asignados a Cara: 0.1
# 2. Recursos asignados a Cara: 0.2    
# 3. Recursos asignados a Cara: 0.3
# 4. Recursos asignados a Cara: 0.4
# 5. Recursos asignados a Cara: 0.5
# 6. Recursos asignados a Cara: 0.6
# 7. Recursos asignados a Cara: 0.7
# 8. Recursos asignados a Cara: 0.8
# 9. Recursos asignados a Cara: 0.9
# 10. Recursos asignados a Cara: 1.0

# %%
respuestas["Diversificación"] = [
0, # 0. Recursos asignados a Cara: 0.0
0, # 1. Recursos asignados a Cara: 0.1
0, # 2. Recursos asignados a Cara: 0.2
0, # 3. Recursos asignados a Cara: 0.3
0, # 4. Recursos asignados a Cara: 0.4
0, # 5. Recursos asignados a Cara: 0.5
0, # 6. Recursos asignados a Cara: 0.6
0, # 7. Recursos asignados a Cara: 0.7
0, # 8. Recursos asignados a Cara: 0.8
0, # 9. Recursos asignados a Cara: 0.9
0, # 10. Recursos asignados a Cara: 1.0
"""
Justifique
""",
]


# %% [markdown]
# ### Apuesta individual
#
# Una casa de apuestas paga 3 por Cara y 1.2 por Sello. La moneda tiene 0.5 de probabilidad de que salga Cara o Sello.
# Nos ofrecen jugar 10000 veces, apostando en cada ocasión todos nuestros recursos, 50% a Cara y 50% a Sello.
# ¿Nos conviene jugar?
# Notar que cuando sale Cara, crecen nuestros recursos 50% (Si teníamos 100, pusimos 50 en Cara y nos pagaron 3*50=150).
# Notar que cuando sale Sello, crecen nuestros recursos -40% (Si teníamos 100, pusimos 50 en Cara y nos pagaron 1.2*50=60).
#
# 0. No
# 1. Sí

# %%
respuestas["Apuesta individual"] = [
0, # 0. No
0, # 1. Sí
"""
Justifique
""",
]


# %% [markdown]
# ### Teoría de utilidad esperada
#
# La teoría de utilidad esperada dice que debemos aceptar una apuesta cuando la utilidad esperada es positiva.
# Supongamos que elegimos una apuesta que nos garantiza crecer 50% cuando sale Cara y caer solo 40% cuando sale Sello (el caso anterior).
# Notar que la esperanza de los recursos es positiva, crece a 5% por paso temporal ((150+60)/2=210/2=105).
# Supongamos que nos ofrecen jugar 10000 veces en un instante, usando absolutamente todos los recursos.
# ¿No conviene jugar?
#
# 0. Siempre
# 1. A veces
# 2. Nunca

# %%
respuestas["Teoría de Utilidad Esperada"] = [
0, # 0. Siempre
0, # 1. A veces
0, # 2. Nunca
"""
Justifique
""",
]


# %% [markdown]
# ### Fondo común
#
# Supongamos que ya estamos jugando una apuesta que nos garantiza crecer 50% cuando sale Cara y caer solo 40% cuando sale Sello.
# ¿Nos conviene juntarnos con alguien, y al final de cada paso temporal poner todos los recursos en un fondo común y dividirlos en partes iguales?
#
#
# 0. No conviene
# 1. Indistinto
# 2. Sí conviene

# %%
respuestas["Fondo común"] = [
0, # 0. No conviene
0, # 1. Indistinto
0, # 2. Sí conviene
"""
Justifique
""",
]


# %% [markdown]
# ### Impuestos
#
# Supongamos que ya estamos jugando una apuesta que nos garantiza crecer 50% cuando sale Cara y caer solo 40% cuando sale Sello, y ya estamos hace mucho tiempo jugando con un fondo común de en un grupo de 100 personas.
# ¿Qué pasa con la tasa de crecimiento de nuestros recursos si logramos encontrar la forma de dejar de aportar nuestra cuota al fondo común mientras tenemos garantizado que seguimos recibimos la cuota del fondo en partes iguales?
#
# 0. Disminuye
# 1. No cambia
# 2. Aumenta

# %%
respuestas["Impuestos"] = [
0, # 0. Disminuye
0, # 1. No cambia
0, # 2. Aumenta
"""
Justifique
""",
]


