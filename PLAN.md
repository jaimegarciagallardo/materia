# El Guardián: informe de evaluación y plan del equipo (P13, P, D, J)

## Contexto
Hay que sustituir `kit/ejemplo_bot.py` por `kit/mi_bot.py`: un detector que devuelve un score de alucinación en [0,1], con el corte en 0,5. Coste: 4·escapes + falsas alarmas. Al dar 1 a todo se obtiene 0, así que el reto real es **dejar pasar como fieles solo los casos con P(alucina) < 0,2**. En esta sesión solo se ha leído y medido; no se ha modificado ningún fichero.

---

## FASE 1: Evaluación

### 1.1 Bot de ejemplo
Para cada oración toma `max(respaldo_por_fuente)` y devuelve `1 − min(oraciones)`. Falla por tres motivos:
- **Compara con cada fuente por separado.** Una respuesta fiel que combina las fuentes 2 y 3 no la respalda ninguna fuente entera, así que da falsa alarma.
- **No usa el juez LLM**, que es la mejor señal con diferencia (ver 1.3).
- **El corte no está calibrado.** Los scores se concentran cerca de 0 o de 1, y el mejor corte en H es 0,003, que en la práctica equivale a «marcarlo todo».

### 1.2 Línea base (medida con `evaluar_local.py`)
| Conjunto | n (aluc.) | Puntuación @0,5 | AUROC | FP | FN | Mejor corte → puntuación |
|---|---|---|---|---|---|---|
| Train completo | 1707 (890) | **0,024** | 0,802 | 381 | 104 | 0,167 → 0,155 |
| `--prefijo H` | 631 (297) | **−0,443** | 0,676 | 170 | 78 | 0,003 → 0,027 |
| `--solo-escenarios` | 1262 | −0,268 | 0,697 | 367 | 102 | 0,015 → 0,033 |

Fuera de H el problema es fácil. En H, el bot de ejemplo es peor que no hacer nada.

### 1.3 Qué señales sirven (AUROC de cada señal por separado, en H frente al resto)
| Señal (peor oración o fragmento) | AUROC H | AUROC no-H |
|---|---|---|
| **LLM juez** (veredicto × confianza) | **0,809** | 0,979 |
| max `respaldo_por_fuente` (oración) | 0,676 | 0,859 |
| min `respaldo_todas` (oración) | 0,605 | 0,905 |
| min max(fuente, todas) (fragmento) | 0,661 | 0,902 |
| respaldo de la fuente que cita el LLM | 0,647 | 0,797 |
| `contradiccion_*` | ~0,48 (**ruido**) | 0,74 |
| `similitud_por_fuente` | 0,42 (**engaña**, invertida) | 0,65 |

**Tabla cruzada del LLM en H** (631 casos): «respaldada» → 324 fieles y **178 alucinaciones** (95 distorsiones, 64 sobregeneralizaciones, 17 no respaldadas). «contradicha» o «no_respaldada» → 10 fieles y 118 alucinaciones. Conclusión: cuando el LLM dice que no, casi siempre acierta (92 %). Cuando dice «respaldada», en H todavía hay un 35 % de alucinaciones, y para dejar pasar un caso hay que bajar del 20 %.

**Ejemplos concretos de train (resumidos):**
- **H201::r4** (distorsión): la respuesta añade «desde 2027 cobrará 3 € al mes…». El LLM dice «respaldada» con confianza 0,95, pero el NLI da 0,0 de respaldo a la fuente citada. El NLI acierta y el LLM se fía demasiado.
- **H405::r2** (distorsión): la fuente dice «estima… *podría* bajar» y la respuesta dice «prevé… bajará». Desaparece la cautela. LLM: 1,0 «respaldada».
- **H205::r2 / H306::r2** (sobregeneralización): la fuente dice «salvo que lo confirme por videollamada» o «únicamente en el Juzgado n.º 3». La respuesta quita la restricción. El NLI da 0,97 y 0,85 de respaldo, así que **las dos señales fallan**.
- **H218::r8** (no respaldado): añade «para agilizar la tramitación», una causa inventada. LLM: «respaldada».
- **Primera oración de H (fiel)**: `respaldo_todas` = 0,001 con una sola fuente a 0,993. `respaldo_todas` se hunde con 8 fuentes concatenadas y no sirve por sí solo en H.

**Reglas léxicas probadas sobre el subconjunto H en que el LLM dice «respaldada»** (tasa de alucinación cuando la regla se activa y cuando no):
- desaparece un matiz de la fuente citada (`salvo`, `únicamente`, `solo`, `podría`, `estima`, `más de`, `al menos`, `algunos`…): **0,65** frente a 0,29. **Es la mejor regla.**
- se añade un conector causal (`para`, `gracias a`, `debido a`, `porque`): 0,43 frente a 0,35. Señal débil.
- aparece una cifra que no está en las fuentes: 0,27 frente a 0,40. **Funciona al revés**: las respuestas fieles hacen cuentas (sumas, «el doble»). Comprobar solo si la cifra existe es contraproducente, hay que comprobar la aritmética.
- «limpio» (fuente citada con respaldo > 0,9, sin matiz perdido, sin causa añadida, sin cifra nueva): 72 casos, **17 % de alucinaciones**, por debajo del 20 %, así que se pueden dejar pasar. Los escapes que quedan son sobregeneralizaciones.

**Techo aproximado de una combinación de señales** (regresión logística en numpy, validación cruzada agrupada por caso base `H123` en 5 bloques sobre H):
- entrenar solo con no-H y probar en H: puntuación 0,066 en el mejor corte (**no transfiere**);
- validación cruzada agrupada dentro de H: AUROC 0,819, **puntuación 0,16–0,18**;
- añadir no-H al entrenamiento lo empeora (0,087).
→ Las señales solas dan unos 0,17 en H. El margen restante tiene que venir de **comprobaciones en Python** (matices, alcance, aritmética, atribución).

### 1.4 Riesgos
- **Fuga de datos por grupos.** Los ids `H201::r1…r9` comparten las mismas fuentes. Una validación cruzada aleatoria por caso filtra información, así que **hay que agrupar por la parte del id anterior a `::`** (88 grupos en H). Calibrar el umbral en los mismos casos en que se entrena también infla la nota.
- **Sobreajuste.** Solo hay 88 grupos en H. Las listas de palabras deben ser genéricas (fenómenos lingüísticos) y no vocabulario de las fuentes. Las fuentes del test no aparecen en el train (0 de 3652), así que memorizar no sirve.
- **Dependencia del formato.** En H y en el test hay ~1 oración y ~2 fragmentos por caso. El giro trae **respuestas más largas** y textos técnicos o legales, con más oraciones, así que agregar por la *peor* oración es obligatorio. Hay que tolerar `llm` ausente, `veredicto: null`, `fuente: null` o índices fuera de rango, `fragmentos` vacío y listas vacías. Los casos no-H con varias oraciones (e, s, T…) sirven para simular el giro.
- **Entorno.** **scikit-learn no está instalado** en esta máquina (numpy 2.2.4 sí, Python 3.13). Solo se entrega `mi_bot.py`: **los coeficientes van incrustados en el código** y el bot no lee `train.jsonl` al ejecutarse, porque en la reproducción ante el jurado puede no estar.
- **CSV.** Generarlo siempre con `generar_predicciones.py`. Ese script **no recorta** el score, así que el bot tiene que devolver un `float` en [0,1] sin NaN. Si salta una excepción, el bot debe devolver un valor por encima de 0,5 (fallo seguro: un escape cuesta 4). Comprobar 440 filas (660 tras el giro), sin ids repetidos y con punto decimal. No abrir el CSV con Excel.

**Dudas del reglamento (no las doy por resueltas):**
- (a) El enunciado puntúa al jurado sobre **40** (método 15, calibración y giro 10, originalidad 5, código 5, propuesta de generalización 5), no sobre 30 como en vuestro resumen.
- (b) No queda claro si en la reproducción habrá scikit-learn disponible. Mejor evitarlo.
- (c) No queda claro si se puede reentrenar o recalibrar *con* `giro_publico.jsonl` (no trae etiquetas). Recalibrar el corte mirando la distribución de scores del giro roza el «ajuste al test». Conviene preguntarlo a la organización.

---

## FASE 2: Estrategia

### Alternativas
| | Idea | Señales | Coste | Riesgo de sobreajuste | Mejora estimada en H |
|---|---|---|---|---|---|
| **A. Peor frase con LLM y NLI** | Score de la peor oración: veredicto LLM negativo → alto; «respaldada» → riesgo base que se reduce solo si hay buen respaldo NLI (de la fuente citada, la mejor fuente o el conjunto, por fragmento) y confianza alta | llm, respaldo_por_fuente, respaldo_todas, fragmentos | Bajo (1–2 h) | Bajo | de −0,44 a ~0,10–0,15 |
| **B. Comprobaciones lingüísticas** | Detectar que desaparece un matiz o una restricción (`salvo`, `solo`, `podría`, `más de`), que se endurece un verbo (estima→prevé, propuso→aprobó), que se amplía el alcance (`todos`, plural frente a «n.º 3»), que se añade una causa (`gracias a`, `para`), aritmética con cifras (una cifra nueva vale si sale de sumar, restar, «el doble» o un porcentaje) y atribución (cifra junto a una entidad distinta en la fuente) | texto, fuentes, fuente citada por el LLM | Medio-alto (3–4 h) | **Medio** (las listas de palabras pueden sobreajustarse) | +0,05 a +0,15 sobre A |
| **C. Clasificador numpy** | Regresión logística (o árbol pequeño) sobre las características de A y B, con validación cruzada agrupada en H, ponderada 4:1, coeficientes incrustados | todas | Medio (2 h) | Medio (88 grupos) | ~0,17 solo con señales; ~0,20–0,30 con las de B |
| **D. Calibración del umbral** | Transformación monótona por tramos para que el corte óptimo, elegido con validación cruzada fuera de muestra, quede en 0,5, con margen conservador | salida de A, B o C | Bajo (30 min) | Bajo si se elige con validación cruzada | Imprescindible: sin ella, una buena ordenación puntúa por debajo de 0 |

### Recomendación: **A + B en forma de características → C con regresión logística transparente → D**
Lo justifican los datos:
- el LLM es la columna vertebral (AUROC 0,81) y casi nunca se equivoca cuando dice que no;
- el problema está en los «respaldada» falsos (178 de 297), justo donde las reglas de B tienen señal (matiz perdido: 0,65);
- la combinación con pesos aprendidos a mano no escala, y la regresión logística da pesos explicables al jurado («método sólido, no un montón de reglas frágiles»);
- D convierte la ordenación en puntos.

**Diseño:** `score = sigmoide(w · características)`, después `recalibrar(score, corte_cv) → 0,5`. Si falla el modelo, la regla A sirve de respaldo.

### Reparto (PROVISIONAL: faltan los perfiles)
Hay módulos separados en `kit/guardian/` para no pisarse. Al final se fusionan en un único `mi_bot.py`, porque **solo se entrega un fichero**.

| Persona | Módulo y ficheros | Interfaz |
|---|---|---|
| **P13**: afirmaciones y normalización | `kit/guardian/texto.py` | `normalizar(t)->str`, `cifras(t)->list[float]` (incluye «mil novecientos…», %, unidades, «1.800»), `entidades(t)->set`, `oraciones_peores(caso)->list[dict]` (une cada oración con sus fragmentos y la fuente citada; robusto a campos ausentes) |
| **P**: comprobaciones lingüísticas (B) | `kit/guardian/reglas.py` | `caracteristicas_reglas(oracion:str, fuente_citada:str, fuentes:list[str]) -> dict[str,float]` (matiz_perdido, verbo_endurecido, alcance_ampliado, causa_añadida, cifra_no_derivable, atribucion_cruzada) |
| **D**: señales, modelo y calibración (A, C, D) | `kit/guardian/senales.py`, `kit/herramientas/entrenar.py` (validación cruzada agrupada, imprime coeficientes), `kit/herramientas/informe.py` (resultados de H, no-H y varias oraciones; errores por tipo) | `caracteristicas_senales(caso)->dict[str,float]`; el entrenamiento escribe `COEF`, `MEDIA`, `DESV` y `CORTE` como literales de Python |
| **J**: integración, pruebas, entregas y pitch | `kit/mi_bot.py` (la versión fusionada), `kit/herramientas/fusionar.py` (concatena los módulos en un único `mi_bot.py`), `kit/herramientas/comprobar_csv.py`, `entregas/<equipo>/entrega_N/` | `Guardian.score(caso)` con try/except que devuelve 0,6, recorte a [0,1] y sin NaN; prueba con casos raros (sin `llm`, sin fragmentos, 10 oraciones) |

Reglas de convivencia: cada persona trabaja solo en sus ficheros, en ramas cortas. Las características se devuelven siempre como `dict` con claves en minúscula y valores en [0,1]. Los cambios de interfaz se avisan en el canal del equipo.

**Para cerrar el reparto necesito saber de cada persona:** soltura con Python y numpy, experiencia con expresiones regulares y procesamiento de texto, conocimientos de estadística y validación cruzada, quién tiene cuenta de GitHub y manejo de git, quién hará el pitch, y el nombre de equipo que os ha asignado la organización.

### Plan por horas (jueves 15/10; ahora son unas 11:30)
| Hora | Hito |
|---|---|
| 11:30–12:00 | J: esqueleto de `mi_bot.py` con la regla A y su corte. Todos: leer 10 casos H de cada tipo |
| **12:00** | **Hito 1: bot A > 0 en H**; P13, P y D arrancan sus módulos |
| 12:00–13:30 | P13: `texto.py`. P: matiz, causa y alcance. D: características de señales y validación cruzada agrupada. J: herramientas de fusión y CSV |
| **13:30** | **Entrega 1** (A + D calibrado), aún con 2 de reserva |
| 13:30–14:15 | P: aritmética y atribución. D: regresión logística con B; informe de errores por tipo |
| 14:15–15:00 | Comida |
| **15:00** | **Hito 2: A+B+C+D, validación cruzada en H ≥ 0,20**; prueba con varias oraciones en no-H. **Entrega 2** si mejora |
| **15:30** | **Giro**: `git pull`. J comprueba el formato del giro (campos, nº de oraciones, nº de fuentes) y que el bot no falla. Revisar la distribución de scores (sin etiquetar nada). Arreglar solo robustez, no reglas a medida |
| 16:00–17:30 | Ajustes genéricos (agregación en respuestas largas, verbos legales o técnicos genéricos). **Entrega 3** (primera post-giro) |
| 17:30–18:30 | Congelar código. Ensayar la reproducción en un entorno limpio (solo `mi_bot.py` + `generar_predicciones.py`). **Entrega 4** |
| **18:30–18:50** | **Entrega final** (5 o 6). Margen de 10 minutos para PR, formato y sincronización del fork |
| 19:00 | Cierre |

---

## FASE 3: Comparativa y decisión

| Criterio | A | B | C | D | **A+B→C+D** |
|---|---|---|---|---|---|
| Puntuación esperada en H | 0,10–0,15 | sola: baja | 0,17 (solo señales) | multiplicador | **0,20–0,30** |
| Robustez al giro | Alta | Media (texto legal) | Media | Alta | Media-alta |
| Complejidad | Baja | Media-alta | Media | Baja | Media |
| Tiempo | 1 h | 3–4 h | 2 h | 0,5 h | ~6 h en paralelo |
| Facilidad de explicarlo (método, calibración, originalidad) | Media | Alta (originalidad) | Alta (pesos legibles) | Alta (calibración) | **Alta** |

**Implementar primero:** A + D (es la red de seguridad y el hito 1). Después, matiz perdido o restricción eliminada y causa añadida (la mejor señal medida), y la regresión logística con validación cruzada agrupada.
**Como mejora si hay tiempo:** aritmética para cifras derivadas, atribución cifra–entidad, endurecimiento de verbos, árbol pequeño en vez de regresión logística.
**No hacer:** usar scikit-learn en el bot ni leer `train.jsonl` en tiempo de ejecución; validación cruzada sin agrupar; calibrar sobre todo el train o con no-H; reglas tipo «cifra no presente → alucina» (funcionan al revés en H); usar la longitud, los superlativos o las negaciones; subir variantes para «afinar» con el leaderboard; abrir el CSV en Excel.

## Verificación
- `python evaluar_local.py --bot mi_bot.py --datos train.jsonl --prefijo H`, con el objetivo de dejar el mejor corte cerca de 0,5 y la puntuación de validación cruzada por encima de 0,20 (fuera de muestra, desde `herramientas/entrenar.py`).
- La misma medición sin `--prefijo` y con casos de varias oraciones, para comprobar que no se desploma (proxy del giro).
- `generar_predicciones.py` sobre test (y giro) y después `comprobar_csv.py`: número de filas, ids únicos, rango [0,1], punto decimal.
- Prueba en limpio: copiar solo `mi_bot.py` y los scripts del kit a una carpeta temporal y ejecutarlos (es lo que hará el jurado).
