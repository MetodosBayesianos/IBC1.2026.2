# IBC1.2026.2

Materia Inferencia Bayesiana Causal 2026, 2do cuatrimestre, se imparte como electiva en las licenciaturas y doctorados de la Escuela de Ciencia y Tecnología de la Universidad Nacional de San Martín.
Además, se ofrece como curso para la comunidad [Bayes Plurinacional](https://bayesplurinacional.org/).

0. [Introducción y objetivos](#objetivos)
0. [Programa y materiales](#programa)

---

<a name="objetivos"></a>
## Introducción y objetivos

**Todas las ciencias con datos desarrollan argumentos causales para explicar y predecir el mundo**.
La evaluación de hipótesis causales atrae cada vez más el interés de las industrias, que necesitan medir el impacto real de sus acciones.
La ventaja de los modelos causales radica en su **capacidad predictiva, que se adapta naturalmente** a los cambios del contexto como son las intervenciones humanas.
Además sabemos que si los argumentos causales se corresponden con la realidad causal subyacente ningún modelo de inteligencia artificial, por más complejo que sea, puede mejorar su desempeño.

En este curso revisaremos los fundamentos de la **evaluación de modelos causales alternativos** $M$ dado los datos $D$, $P(M|D)$, tanto con y sin intervenciones.
Además revisaremos los métodos para hacer **predicciones de las hipótesis $H$ internas a los modelos $M$**, $P(H|D,M)$, tanto para evaluar efectos causales out-of-sample como para predecir el impacto de acciones alternativas contrafactuales.
Finalmente abordaremos el problema de **tomar decisiones óptimas como un problema de inferencia** en el que se revisa que ninguna de las decisiones contrafactuales alternativas mejoren el resultado de la variable objetivo.

Ante el vértigo de una IA plagada de herramientas efímeras, **este curso prioriza los fundamentos inmutables**.
A pesar de todos los avances, desde el siglo 18 hasta ahora no se ha propuesto un nuevo sistema para razonar bajo incertidumbre.
Si algún día las máquinas superan a los humanos, deberán hacer ciencia aplicando estrictamente las reglas de probabilidad para evaluar teorías causales.
Para alcanzar verdades (intersubjetivas) en contextos de incertidumbre, simplemente hay que saber aplicar y preservar el **principio universal de no mentir**: no afirmar más de lo que se sabe, sin ocultar lo que sí se sabe.
Así se obtienen las **distribuciones de creencia óptima dada la información disponible**, inmejorables en términos prácticos.

Al final siempre es bueno recordar que **la verdadera inteligencia** no es ni artificial ni humana, sino que **está en todas las formas de vida** que son capaces de sobrevivir en el tiempo.
Especialmente las plantas que son 83% de la biomasa, no molestan a nadie y dan vida.
El problema real detrás de los problemas de conocimiento es responder preguntas **¿qué acciones nos generan bienestar?**.

</p>

<a name="programa"></a>
## Programa y materiales

Los contenidos completos del programa se encuentran en [`programa.pdf`](https://github.com/MetodosBayesianos/IBC1.2026.2/blob/main/programa.pdf).

### Unidad 0. Previa

*Materiales*:

* [Presentación](https://github.com/MetodosBayesianos/IBC1.2026.2/tree/main/0-previa/0-previa.pdf)
* [Cuestionario](https://github.com/MetodosBayesianos/IBC1.2026.2/tree/main/0-previa/cuestionario0.py)


### Unidad 1. Especificación y evaluación de argumentos causales.

#### 1.1. Argumentos causales alternativos e incertidumbre

*Materiales*:

* [Video](https://youtu.be/5pzmCWPaRMM?si=qDESYdtz3q6Z9F-z)
* [Teórica](https://github.com/MetodosBayesianos/IBC1.2026.2/tree/main/1.1-argumentos_causales/teorica/1.1-argumentos_causales_e_incertidumbre.pdf)

*Bibliografía* (link en `programa.pdf`):

* Teórica: Capítulos 2 y 3 (hasta el final de la sección 3.2) de libro de Daphne Koller (2009) *Probabilistic Graphical Models*
* Práctica: Capítulo 2 del libro de McElreath (2020) *Statistical rethinking* y capítulo 2 del libro de Winn (2023) *Model Based Machine Learning*

#### 1.2 Sorpresa: el problema de la comunicación con la realidad.

*Materiales*:

* [Video](https://youtu.be/2K9h6mB-xfc?si=DinZY2w5_EcwkQtm)
* [Teórica](https://github.com/MetodosBayesianos/IBC1.2026.2/blob/main/1.2-sorpresa_comunicacion_realidad/teorica/1.2-sorpresa_comunicacion_realidad.pdf)

*Bibliografía* (link en `programa.pdf`):

* Teórica: Secciones 1.1, 2.4-6, 4.1 del libro de MacKay (2003) *Information theory, inference and learning algorithms*
* Práctica: Capítulo 3 y 10 del libro de McElreath (2020) *Statistical rethinking*.


### Unidad 2. Métodos de inferencia y programación probabilística.

#### 2.1. Inferencia exacta y pasaje de mensajes.

*Materiales*:

* [Video](https://youtu.be/zfMJbwcFBjQ?si=t4VjLQ72MGlKyg45)
* [Teórica](https://github.com/MetodosBayesianos/IBC1.2026.2/blob/main/2.1-inferencia_exacta/teorica/2.1-exacta_pasaje_de_mensajes.pdf)

*Bibliografía* (link en `programa.pdf`):

Teórica:

- Koller 2009. Probabilistic Graphical Models. Lectura: cap 9 y 10.
- Bishop 2006. Pattern Recognition and Machine Learning. Lectura: cap 3 (y 2).

Práctica:

- McElreath 2020. Statistical Rethinking. Lectura: cap 4.

#### 2.2. Olvido: hacer inferencia en realidades complejas.

*Materiales*:

* [Video 1](https://drive.google.com/file/d/15JYl7i3K9OZm2hgE6K-MJht7yH-9QYeU/view)
* [Video 2](https://drive.google.com/file/d/1er02P-VIreHd1KxHfkmXFrqBOPxQVhHa/view)
* [Teórica](https://github.com/MetodosBayesianos/IBC1.2026.2/blob/main/2.2-inferencia_aproximada/teorica/)

*Bibliografía*

Teórica:
* Bishop 2006. Pattern Recognition and Machine Learning. Capítulo 10.
* Capítulo 9 del libro de McElreath (2020) *Statistical rethinking*.
* [Simulated-Based inference: A Practical Guide](https://arxiv.org/pdf/2508.12939)

Práctica:
* [Pyro][https://pyro.ai]
* [Trunglang][https://turinglang.org/]
* [PyMC][https://www.pymc.io]
* [Stan][https://mc-stan.org/]

### Trabajo práctico, tres casos reales.

*Materiales*:

* [Modelos para casos reales](https://github.com/MetodosBayesianos/IBC1.2026.2/tree/main/2.2-inferencia_aproximada/practica)

### Unidad 3. Predicciones causales.

#### 3.1. Flujo de inferencia y eliminación de la asociación espuria.

* [Teórica](https://github.com/MetodosBayesianos/IBC1.2026.2/tree/main/3.1-flujo_de_inferencia/teorica)

*Bibliografía*

Teórica:

- Pearl 2009. [Causal inference in statistics: An overview](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf). No estaba en el programa original.

Práctica:

- McElreath 2020. Statistical Rethinking. Lectura: cap 5 y 6.


#### 3.2. Estimandos y do-calculus

12- 16 octubre

Teórica:

- Pearl 2009. [Causal inference in statistics: An overview](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf). No estaba en el programa original.


### Unidad 4. Inferencia causal.

#### 4.1. El zoológico de algoritmos.

20 Octubre

#### 4.2. El choque de paradigmas causales.

27 octubre

Teórica:

- Pearl 2009. [Causal inference in statistics: An overview](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf). No estaba en el programa original.


### Cierre 5.

#### 5.1 Corrección temporal de la teoría de juegos.

03 Noviembre

#### 5.2 Evaluación final.

10 Noviembre



























