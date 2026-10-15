# Organización del equipo: quién hace qué

> Reparto **provisional**. Si alguien se siente más cómodo en otra parte, se cambia.
> El detalle técnico está en `PLAN.md`.

## La idea en una frase

El bot mira la **peor frase** de la respuesta. Se fía del juez LLM cuando dice «NO»,
y cuando dice «sí» le pasamos **nuestras comprobaciones de texto** para pillar lo
que se le escapa. Un modelo sencillo junta todo en un número, que ajustamos para
que el corte quede en 0,5.

## Cómo encajan las piezas

```
                caso (pregunta, fuentes, respuesta, señales)
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │  P13 · texto.py                      │
               │  Prepara: frases, cifras, palabras   │
               │  normalizadas, fuente que cita el LLM│
               └───────┬──────────────────────┬───────┘
                       │                      │
                       ▼                      ▼
       ┌─────────────────────────┐  ┌─────────────────────────┐
       │  P · reglas.py          │  │  D · senales.py         │
       │  ¿Falta un «solo»?      │  │  Juez LLM, NLI,         │
       │  ¿Causa inventada?      │  │  respaldo por fuente,   │
       │  ¿«todos» en vez de     │  │  fragmentos...          │
       │  «algunos»? ¿Cifras?    │  │                         │
       └───────────┬─────────────┘  └────────────┬────────────┘
                   │      números entre 0 y 1    │
                   └──────────────┬──────────────┘
                                  ▼
               ┌──────────────────────────────────────┐
               │  D · modelo + calibración            │
               │  Pesa cada pista y mueve el corte    │
               │  para que caiga en 0,5               │
               └──────────────────┬───────────────────┘
                                  ▼
               ┌──────────────────────────────────────┐
               │  J · mi_bot.py                       │
               │  Junta todo en UN fichero, lo prueba │
               │  y hace las entregas                 │
               └──────────────────┬───────────────────┘
                                  ▼
                          score entre 0 y 1
```

## Lo que hace cada uno

### P13: preparar el texto (`kit/guardian/texto.py`)
Herramientas que usan los demás:
- `normalizar(texto)`: minúsculas, sin tildes ni signos raros.
- `cifras(texto)`: saca los números («1.800», «12 %», «mil novecientos setenta y ocho»).
- `fuente_citada(oracion, fuentes)`: el texto de la fuente en la que se apoya el juez LLM.
- Que **nunca falle**: frases sin `llm`, sin fragmentos, con una fuente que no existe...

**Lo primero:** `normalizar` y `cifras`, porque P las necesita pronto.

### P: comprobaciones de texto (`kit/guardian/reglas.py`)
Busca lo que el juez LLM no ve. Una función que devuelve números entre 0 y 1:
`caracteristicas_reglas(oracion, fuente_citada, fuentes) -> dict`

Por orden de importancia:
1. **Matiz perdido:** la fuente dice «solo», «salvo», «podría», «más de», «algunos»... y la respuesta no. *(Es la mejor pista que hemos encontrado.)*
2. **Causa inventada:** la respuesta añade «para...», «gracias a», «porque»...
3. **Alcance ampliado:** «el Juzgado n.º 3» pasa a «los Juzgados»; «algunos» pasa a «todos».
4. **Verbo más fuerte:** «propuso» pasa a «aprobó»; «estima» pasa a «prevé».
5. *(Si da tiempo)* **Cifras:** comprobar si salen de hacer cuentas con las fuentes.
6. *(Si da tiempo)* **Atribución:** el número existe, pero es de otra entidad.

**Ojo:** solo palabras genéricas del español, nada sacado de casos concretos.

### D: señales, modelo y calibración (`kit/guardian/senales.py` + `kit/herramientas/`)
- `caracteristicas_senales(caso) -> dict`: veredicto y confianza del juez LLM, respaldo del NLI (de la fuente citada, de la mejor fuente, de todas juntas) y peor fragmento.
- Entrena un modelo sencillo con numpy (**sin scikit-learn**) que junta lo de D y lo de P.
- Valida **solo con los casos H** y **agrupando por caso** (H201::r1, r2... van juntos).
- Elige el corte sabiendo que un escape cuesta 4 y lo recoloca en 0,5.
- Escribe los pesos como números dentro del código; el bot no lee `train.jsonl`.

**Lo primero:** una versión rápida «solo juez LLM + NLI» para tener algo que entregar.

### J: integración, pruebas y entregas (`kit/mi_bot.py` + `entregas/`)
- Monta `mi_bot.py` con la clase `Guardian`. **Se entrega un solo fichero**, así que hay que pegar dentro los módulos de los demás.
- Si algo falla dentro del bot, devuelve 0,6 (mejor una falsa alarma que un escape).
- Prueba con casos raros (sin LLM, sin fragmentos, respuestas largas).
- Genera el CSV **siempre con el script** (nunca con Excel) y comprueba: una fila por caso, sin repetidos, punto decimal.
- Hace los Pull Request de las entregas y vigila el giro de las 15:30 (`git pull`).
- Prepara el guion del pitch con lo que cuente cada uno.

## Reglas para no pisarnos

- Cada uno toca **solo sus ficheros**.
- Todos devolvemos un `dict` de números entre 0 y 1 con nombres claros.
- Si cambias lo que devuelve tu función, **avisa al grupo**.
- Cada cambio se mide antes de darlo por bueno:
  `python evaluar_local.py --bot mi_bot.py --datos train.jsonl --prefijo H`

## Horario

| Hora | P13 | P | D | J |
|---|---|---|---|---|
| hasta 12:00 | `normalizar`, `cifras` | leer 10 casos H de cada tipo | versión rápida LLM + NLI | esqueleto de `mi_bot.py` |
| 12:00–13:30 | resto de `texto.py` | matiz perdido + causa inventada | características de señales | juntar módulos + comprobar CSV |
| **13:30** | | | | **Entrega 1** |
| 13:30–14:15 | ayudar a P con cifras | alcance + verbos | modelo + corte | pruebas con casos raros |
| 14:15–15:00 | comida | comida | comida | comida |
| **15:00** | | | | **Entrega 2** (si mejora) |
| **15:30 giro** | revisar formato nuevo | ¿mis reglas aguantan textos legales? | ¿el corte sigue bien? | `git pull` + que no pete |
| 16:00–17:30 | arreglos | arreglos genéricos | ajustes | **Entrega 3** |
| 17:30–18:50 | | **código congelado** | | prueba en limpio + **entrega final** |
| 19:00 | | | | **cierre** |

## Qué NO hacer

- Usar scikit-learn o leer `train.jsonl` dentro del bot.
- Reglas pensadas para casos concretos del test (está prohibido).
- «Si una cifra no está en las fuentes, alucina»: en los casos H funciona al revés.
- Abrir el CSV con Excel.
- Subir variantes solo para ver si sube el leaderboard.
