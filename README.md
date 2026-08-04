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

Los contenidos completos del programa se encuentran en [su carpeta específica](https://github.com/MetodosBayesianos/IBC1.2026.2/blob/main/programa.pdf).

### Unidad 0. Previa

*Materiales*:

* [Presentación](https://github.com/MetodosBayesianos/IBC1.2026.2/tree/main/0-previa.pdf)
* [Cuestionario](https://github.com/MetodosBayesianos/IBC1.2026.2/tree/main/cuestionario0.py)


### Unidad 1. Especificación y evaluación de argumentos causales.

#### 1.1. Argumentos causales alternativos e incertidumbre.

*Materiales*:

* [Video](https://youtu.be/5pzmCWPaRMM?si=qDESYdtz3q6Z9F-z)
* [Teórica](https://github.com/MetodosBayesianos/IBC1.2026.2/tree/main/1.1-argumentos_causales/teorica)

*Bibliografía*:

* Teórica: Capítulos 2 y 3 (hasta el final de la sección 3.2) de libro de Daphne Koller (2009) *Probabilistic Graphical Models*
* Práctica: Capítulo 2 del libro de McElreath (2020) *Statistical rethinking* y capítulo 2 del libro de Winn (2023) *Model Based Machine Learning*

#### 1.2 Sorpresa: el problema de la comunicación con la realidad.

*Bibliografía*:

* Teórica: Secciones 1.1, 2.4-6, 4.1 del libro de MacKay (2003) *Information theory, inference and learning algorithms*
* Práctica: Capítulo 3 y 10 del libro de McElreath (2020) *Statistical rethinking*.



