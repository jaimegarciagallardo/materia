# Parte de P: reglas de verificación

Comprobaciones de texto en Python puro (solo librería estándar) para pillar lo que el
juez LLM deja pasar. **No deciden solas**: devuelven pistas entre 0 y 1 que el modelo
de D combina con las señales.

| Fichero | Para qué |
|---|---|
| `reglas.py` | El módulo. Es lo único que va dentro del bot. |
| `evaluar_reglas.py` | Mide cada regla en H, H con LLM=«respaldada», no-H y test, y lo que aporta al modelo. |
| `probar_reglas.py` | 20 pruebas con frases inventadas y casos raros (sin `llm`, sin fragmentos...). |

Desde la raíz del repositorio (`material/`):

```
python guardian/probar_reglas.py                                                 # ¿funciona todo?
python guardian/evaluar_reglas.py                                                # informe completo
python guardian/evaluar_reglas.py --ejemplos modal_perdido --etiqueta 0 --n 5    # ver falsas alarmas de una regla
```

## Casos H y no-H

- **H** (631 casos, por ejemplo `H201::r4`): están escritos como el test, con unas 8 fuentes, una frase y mentiras sutiles. `H201` es el tema y `r1…r9` son respuestas distintas sobre ese tema (88 temas). **Aquí se mide la nota.** Al validar, las respuestas de un mismo tema van siempre juntas.
- **No-H** (1.076 casos: `e`, `s`, `T`, `G`, `R`, `N`, `g`): son más fáciles y tienen otro formato. No se usaron para diseñar las reglas, así que sirven de **control** y de simulación del giro.

## Mapa de `reglas.py`

| Bloque | Qué hace |
|---|---|
| A. Texto | `normalizar`, `tokens`, `raiz` y `clausulas_relevantes`: quitan tildes, parten en palabras y se quedan con los trozos de las fuentes que hablan de lo mismo que la frase. |
| B. Cifras | `extraer_cifras` entiende «1.800», «2,7», «6,8 millones», «seis de cada diez» y «10:05». `comprobar_cifras` mira si cada cifra está en las fuentes o sale de una cuenta (suma, resta, %, «el resto»). |
| C. Matices | `matices_perdidos` (podría, si, solo, propuso, condicionales) y `aproximacion_perdida`. |
| D. Alcance | `alcance` («14 de los 23» → «los») y `modificador_omitido` («vino embotellado europeo» → «vino europeo»). |
| E. Causas | `conectores_anadidos`: «gracias a», «para…» o «lo que permitió» sin respaldo. |
| F. Contenido nuevo | `contenido_nuevo`: se calcula, pero **no se usa** (es un atajo, ver abajo). |
| G. Interfaz | `reglas_caso`, `caracteristicas_reglas`, `NOMBRES`, `RECOMENDADAS`. |

## Interfaz (para D y J)

```python
from reglas import reglas_caso, caracteristicas_reglas, RECOMENDADAS

pistas = reglas_caso(caso)          # dict {nombre: 0/1 o fracción}; máximo sobre las oraciones
x = [pistas[k] for k in RECOMENDADAS]   # las que hay que meter en el modelo

# o por oración, si D quiere su propia agregación:
caracteristicas_reglas(oracion, fuentes_relevantes, fuentes, pregunta="")
```

- `reglas_caso` elige solo las fuentes relevantes de cada oración: la que cita el LLM, las 2 que más respaldan según el NLI y las 2 más parecidas.
- **Nunca lanza excepciones** en modo normal. Si algo falla, devuelve todo a 0 y el resto de señales sigue funcionando.
- Tarda unos 5 s para los 2.147 casos de train y test.
- Para el `mi_bot.py` final, J puede pegar `reglas.py` entero, porque no tiene dependencias.

## Qué comprueba (las 12 recomendadas)

| Pista | Qué detecta | Ejemplo (frases inventadas) |
|---|---|---|
| `cifra_no_derivable` | Una cifra que ni está en las fuentes ni sale de una cuenta directa (suma, resta, %, «el resto», «seis de cada diez»...) | fuente «12 %» → respuesta «21 %»; «de 200 a 260, un 40 % más» |
| `cifra_dudosa` | La cifra solo «sale» combinándola con una fuente que no habla del tema | |
| `ratio_falso` | «el doble», «triplica»... que las cifras no sostienen | 280 «triplica» 120 |
| `modal_perdido` | Desaparece un «podría» o un condicional | «podría costar» → «costará»; «permitiría» → «permitirá» |
| `condicion_perdida` | Desaparece un «si», «siempre que», «salvo», «pendiente de» | «gratis siempre que estén empadronados» → «gratis» |
| `restriccion_perdida` | Desaparece un «solo», «únicamente» | «únicamente en el juzgado n.º 3» → «en los juzgados» |
| `intencion_como_hecho` | Una propuesta o petición se cuenta como hecho | «propuso subir la tasa» → «subirá la tasa» |
| `aproximacion_perdida` | «más de 150.000» → «150.000» | |
| `parcial_perdido` | «14 de los 23 comedores» → «los comedores» | |
| `modificador_omitido` | Se cae un adjetivo o complemento que acota | «vino embotellado europeo» → «vino europeo» |
| `causa_anadida` | Una causa o finalidad que ninguna fuente da | «gracias a una donación», «para agilizar» |
| `consecuencia_anadida` | «lo que permitió...», «de modo que...» sin respaldo | |

Las cifras se calculan con cuidado:
- `1.800` es mil ochocientos y `2,7` es dos coma siete.
- «6,8 millones» vale 6.800.000, y en «de 6,4 a 6,8 millones» el «millones» se aplica a las dos.
- Las horas cuentan como minutos, así que de «10:05» a «10:23» salen 18 minutos.
- Los años y las fechas solo se comparan tal cual, sin hacer cuentas con ellos.

## Resultados (train, casos H)

Contando todos los casos H (334 fieles y 297 alucinaciones), las reglas buenas saltan sobre todo en alucinaciones. Dentro de los 502 casos en que el LLM dice «respaldada», donde la tasa base es del 35 %, siguen saltando con un 55–90 % de alucinaciones:

| Pista | Fieles en que salta | Alucinaciones en que salta | % alucinación cuando salta |
|---|---|---|---|
| cifra_no_derivable | 4 | 28 | 88 % |
| modal_perdido | 6 | 35 | 85 % |
| parcial_perdido | 7 | 26 | 79 % |
| restriccion_perdida | 10 | 30 | 75 % |
| condicion_perdida | 15 | 42 | 74 % |
| intencion_como_hecho | 5 | 17 | 77 % |
| causa_anadida | 9 | 22 | 71 % |
| modificador_omitido | 9 | 20 | 69 % |

**Lo que aportan al modelo.** Regresión logística ponderada 4:1, validada en H agrupando por caso base (H201::r1…r9 siempre juntos):

| Modelo | AUROC (CV en H) | Mejor puntuación (CV en H) | Entrenado en H → probado en no-H (@0,5) |
|---|---|---|---|
| Solo señales | 0,812 | +0,17 | +0,30 |
| **Señales + reglas recomendadas** | **0,869** | **+0,33** | **+0,54** |
| Señales + todas las pistas | 0,900 | +0,39 | +0,14 ⚠️ |

**Por qué no usar todas.** `palabras_nuevas`, `cifra_nueva`, `universal_anadido`, `cifra_de_otra_fuente`, `racha_nueva` y `fecha_nueva` funcionan *al revés* en H: una respuesta con más palabras o cifras nuevas tiende a ser fiel. Es un artefacto de cómo se escribieron las alucinaciones de H, que son ediciones mínimas de la fuente. En H suben un poco la nota, pero al cambiar de formato la hunden. El giro cambia de formato, así que se calculan pero **no van al modelo**.

## Honestidad y límites (para la defensa)

- Las reglas son **fenómenos generales del español** (los que cita el enunciado: «podría», «solo», «propuso ≠ aprobó», «gracias a», cifras). No hay ninguna palabra sacada de un caso concreto.
- Las diseñé mirando ejemplos de H. La validación cruzada no protege de eso, así que el control honesto es **no-H**, que no usé para diseñarlas: ahí las reglas aciertan entre el 55 % y el 93 % de las veces que saltan, y la mejora se mantiene. Además, la frecuencia con que saltan en el test es parecida a la de H.
- **Lo que se escapa todavía:**
  - Sobregeneralizaciones sin una palabra clave, por ejemplo «Torralcina ganó población» convertido en «los municipios ganaron población».
  - Atribuciones cruzadas: la cifra existe en las fuentes, pero es de otra entidad.
  - Porcentajes que coinciden por casualidad con alguna cuenta.
- **Riesgo en el giro (textos legales):** abundan «salvo», «siempre que» y «podrá». `condicion_perdida` y `modal_perdido` pueden saltar más en respuestas fieles. Hay que revisar a las 15:30 en qué porcentaje del giro saltan.

## Siguientes pasos (si da tiempo)

1. Atribución: comprobar que la entidad que acompaña a la cifra en la respuesta es la misma que la acompaña en la fuente.
2. Verbos endurecidos sin marcador: «vacunó» → «inmunizó», «presupuesto» → «costará».
3. Reutilizar `normalizar` y `cifras` de P13 cuando estén, para no duplicar código. Las de aquí sirven mientras tanto.

> Nota: el reparto en `ORGANIZACION.md` decía `kit/guardian/reglas.py`; el módulo vive en `guardian/`, en la raíz del repositorio.
> El proceso completo, paso a paso, está en `PROCESO_P.md` (fuera del repositorio).
