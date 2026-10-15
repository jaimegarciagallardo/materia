"""Entrena el modelo y escribe kit/mi_bot.py (un solo fichero, sin dependencias).

    cd kit
    python herramientas/montar_bot.py

Qué hace:
  1. Calcula las características (señales de D + reglas de P) de los casos del train.
  2. Entrena con H (peso x3: es lo que más se parece al test) y con las demás
     familias (G, R, N, T, s), que aportan formatos distintos (respuestas largas,
     contratos...) y ayudan a generalizar al giro. La familia «e» (1 fuente de una
     línea) se deja fuera: no se parece ni al test ni al giro.
  3. Validación cruzada AGRUPADA por tema en H (H201::r1..r9 siempre juntos), 5
     bloques × 3 semillas, con las demás familias siempre en el entrenamiento: cada
     caso H se puntúa con un modelo que no lo ha visto. Con eso se elige el corte.
  4. Simulación del giro: se entrena SIN una familia (G o R) y se mide en ella,
     con el corte elegido en H. Es lo más parecido a recibir un formato nuevo.
  5. Respuestas largas (el giro): se simulan juntando 3-4 frases H del mismo tema
     (comparten fuentes): la mitad todas fieles, la otra mitad con UNA frase falsa.
     El bot puntúa cada frase y las combina con noisy-OR; el corte de este caso se
     elige con esas respuestas simuladas, fuera de muestra (agrupando por tema).
  6. Entrena el modelo final; el bot recoloca cada corte en 0,5.
  7. Escribe mi_bot.py = reglas.py + plantilla_bot.py + pesos, y comprueba que
     el fichero escrito da exactamente las mismas puntuaciones.
"""
import collections
import copy
import importlib.util
import json
import math
import os
import random
import re
import sys

import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARDIAN = os.path.join(KIT, "guardian")
sys.path.insert(0, GUARDIAN)
import plantilla_bot as P  # noqa: E402

PESO_ESCAPE = 4.0   # un escape pesa 4 veces más que una falsa alarma
PESO_H = 3.0        # los casos H se parecen al test: pesan más
FAMILIAS = ["H", "G", "R", "N", "T", "s"]
L2 = 1.0


def cargar(nombre):
    with open(os.path.join(KIT, nombre), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def puntuacion(y, p, t):
    fn = np.sum((y == 1) & (p < t))
    fp = np.sum((y == 0) & (p >= t))
    return 1 - (4 * fn + fp) / np.sum(y == 0), int(fn), int(fp)


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / len(pos) / len(neg)


def entrenar(X, y, pesos=None):
    media, desv = X.mean(0), X.std(0)
    desv[desv < 1e-9] = 1.0
    Z = np.c_[(X - media) / desv, np.ones(len(X))]
    w = np.where(y == 1, PESO_ESCAPE, 1.0) * (1.0 if pesos is None else pesos)
    b = np.zeros(Z.shape[1])
    for _ in range(3000):
        p = 1 / (1 + np.exp(-Z @ b))
        g = Z.T @ ((p - y) * w) / w.sum() + L2 * np.r_[b[:-1], 0] / len(y)
        b -= 0.5 * g
    return {"media": media, "desv": desv, "coef": b[:-1], "sesgo": b[-1]}


def predecir(m, X):
    z = ((X - m["media"]) / m["desv"]) @ m["coef"] + m["sesgo"]
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def familia(c):
    return c["id"][0]


def respuestas_largas(train, por_tema=6, semilla=0):
    """Respuestas de 3-4 frases simuladas con frases H del mismo tema (mismas fuentes).
    Etiqueta 0: todas fieles. Etiqueta 1: exactamente una frase falsa entre fieles."""
    rng = random.Random(semilla)
    por = collections.defaultdict(list)
    for c in train:
        if familia(c) == "H" and len(c["senales"]["oraciones"]) == 1:
            por[c["id"].split("::")[0]].append(c)
    casos = []
    for tema, cs in sorted(por.items()):
        fieles = [c for c in cs if c["etiqueta"] == 0]
        falsas = [c for c in cs if c["etiqueta"] == 1]
        if len(fieles) < 3 or not falsas:
            continue
        for i in range(por_tema):
            k = rng.choice((3, 4))
            if i % 2 == 0:
                partes, etiqueta = rng.sample(fieles, min(k, len(fieles))), 0
            else:
                partes, etiqueta = rng.sample(fieles, min(k - 1, len(fieles))) + [rng.choice(falsas)], 1
            rng.shuffle(partes)
            oraciones, fragmentos = [], []
            for c in partes:
                base = len(oraciones)
                oraciones += copy.deepcopy(c["senales"]["oraciones"])
                for f in c["senales"]["fragmentos"]:
                    fragmentos.append(dict(copy.deepcopy(f), oracion=base + int(f.get("oracion", 0))))
            casos.append({"id": f"{tema}::L{i}", "pregunta": partes[0]["pregunta"], "fuentes": partes[0]["fuentes"],
                          "respuesta": " ".join(c["respuesta"].strip() for c in partes),
                          "senales": {"oraciones": oraciones, "fragmentos": fragmentos}, "etiqueta": etiqueta})
    return casos


def probs_largas(m, nombre, frases_d):
    """noisy-OR: probabilidad de que al menos una frase mienta."""
    ps = predecir(m, np.array([P.vector(d, nombre) for d in frases_d], float))
    return 1 - np.prod(1 - ps)


def validar_larga(datos, largas, nombre, corte_unica):
    """Elige el corte de las respuestas largas fuera de muestra (agrupando por tema)."""
    X = np.array([P.vector(d, nombre) for _, d in datos], float)
    y = np.array([c["etiqueta"] for c, _ in datos], float)
    fam = np.array([familia(c) for c, _ in datos])
    tema = np.array([c["id"].split("::")[0] for c, _ in datos])
    pesos = np.where(fam == "H", PESO_H, 1.0)
    yL = np.array([c["etiqueta"] for c, _ in largas], float)
    temaL = np.array([c["id"].split("::")[0] for c, _ in largas])
    oof = []
    for semilla in (0, 1, 2):
        temas = np.unique(temaL)
        np.random.default_rng(semilla).shuffle(temas)
        Pr = np.zeros(len(yL))
        for bloque in np.array_split(temas, 5):
            fuera = np.isin(tema, bloque) & (fam == "H")
            m = entrenar(X[~fuera], y[~fuera], pesos[~fuera])
            for i in np.where(np.isin(temaL, bloque))[0]:
                Pr[i] = probs_largas(m, nombre, largas[i][1])
        oof.append(Pr)
    cortes = np.round(np.arange(0.30, 0.961, 0.01), 2)
    medias = [np.mean([puntuacion(yL, Pr, t)[0] for Pr in oof]) for t in cortes]
    corte = float(cortes[int(np.argmax(medias))])
    fn = np.mean([puntuacion(yL, Pr, corte)[1] for Pr in oof])
    fp = np.mean([puntuacion(yL, Pr, corte)[2] for Pr in oof])
    sin = np.mean([puntuacion(yL, Pr, corte_unica)[0] for Pr in oof])
    giro = []
    for nueva in ("G", "R"):
        dentro = fam != nueva
        m = entrenar(X[dentro], y[dentro], pesos[dentro])
        cs = [(c, d) for c, d in datos if familia(c) == nueva]
        p, t = [], []
        for c, d in cs:
            partes = [P.caracteristicas(x) for x in P.frases(c)]
            if len(partes) > 1:
                p.append(probs_largas(m, nombre, partes)), t.append(corte)
            else:
                p.append(predecir(m, np.array([P.vector(d, nombre)], float))[0]), t.append(corte_unica)
        yy, p, t = np.array([c["etiqueta"] for c, _ in cs], float), np.array(p), np.array(t)
        efn, efp = int(((yy == 1) & (p < t)).sum()), int(((yy == 0) & (p >= t)).sum())
        giro.append(f"{nueva} {1 - (4 * efn + efp) / (yy == 0).sum():+.2f} (esc {efn}, FA {efp})")
    print(f"  {nombre:9s} largas simuladas ({len(yL)}): corte {corte:.2f} -> {max(medias):+.3f} (escapes {fn:.0f}, "
          f"falsas alarmas {fp:.0f}; con el corte de frase única {sin:+.3f})  |  familia nueva: {'  '.join(giro)}")
    return corte


def validar(datos, nombre):
    """CV agrupada en H (las demás familias siempre entrenan) + simulación del giro.
    Devuelve el corte elegido fuera de muestra en H."""
    X = np.array([P.vector(d, nombre) for _, d in datos], float)
    y = np.array([c["etiqueta"] for c, _ in datos], float)
    fam = np.array([familia(c) for c, _ in datos])
    tema = np.array([c["id"].split("::")[0] for c, _ in datos])
    pesos = np.where(fam == "H", PESO_H, 1.0)
    esH = fam == "H"
    oof = []
    for semilla in (0, 1, 2):
        temas = np.unique(tema[esH])
        np.random.default_rng(semilla).shuffle(temas)
        Pr = np.zeros(esH.sum())
        for bloque in np.array_split(temas, 5):
            fuera = np.isin(tema, bloque) & esH
            m = entrenar(X[~fuera], y[~fuera], pesos[~fuera])
            Pr[np.isin(tema[esH], bloque)] = predecir(m, X[fuera])
        oof.append(Pr)
    yH = y[esH]
    cortes = np.round(np.arange(0.05, 0.951, 0.01), 2)
    medias = [np.mean([puntuacion(yH, Pr, t)[0] for Pr in oof]) for t in cortes]
    corte = float(cortes[int(np.argmax(medias))])
    fn = np.mean([puntuacion(yH, Pr, corte)[1] for Pr in oof])
    fp = np.mean([puntuacion(yH, Pr, corte)[2] for Pr in oof])
    giro = []
    for nueva in ("G", "R"):
        dentro = fam != nueva
        m = entrenar(X[dentro], y[dentro], pesos[dentro])
        s, efn, efp = puntuacion(y[~dentro], predecir(m, X[~dentro]), corte)
        giro.append(f"{nueva} {s:+.2f} (esc {efn}, FA {efp})")
    print(f"  {nombre:9s} H: AUROC {np.mean([auroc(yH, Pr) for Pr in oof]):.3f} · corte {corte:.2f} -> "
          f"{max(medias):+.3f} (escapes {fn:.0f}, falsas alarmas {fp:.0f})  |  familia nueva: {'  '.join(giro)}")
    return corte


def main():
    train = cargar("train.jsonl")
    datos = [(c, P.caracteristicas(c)) for c in train if familia(c) in FAMILIAS]
    H = [c for c, _ in datos if familia(c) == "H"]
    print(f"Entrenamiento: {len(datos)} casos ({', '.join(f'{f} {sum(familia(c) == f for c, _ in datos)}' for f in FAMILIAS)}),"
          f" H pesa x{PESO_H:g}.")
    print("Validación: CV agrupada en H (5 bloques × 3 semillas) y «familia nueva» (entrenar sin G o sin R y medir en ella):")
    modelos = {}
    pesos = np.array([PESO_H if familia(c) == "H" else 1.0 for c, _ in datos])
    y = np.array([c["etiqueta"] for c, _ in datos], float)
    cortes = {}
    for nombre in P.MODELOS:
        cortes[nombre] = validar(datos, nombre)
    largas = [(c, [P.caracteristicas(x) for x in P.frases(c)]) for c in respuestas_largas(train)]
    print(f"Respuestas largas (como el giro), simuladas con frases H del mismo tema; puntuación por frase + noisy-OR:")
    for nombre in P.MODELOS:
        corte_larga = validar_larga(datos, largas, nombre, cortes[nombre])
        m = entrenar(np.array([P.vector(d, nombre) for _, d in datos], float), y, pesos)
        m["corte_unica"], m["corte_larga"] = cortes[nombre], corte_larga
        modelos[nombre] = m
    print("\nPesos del modelo completo (estandarizados; positivo = empuja a «alucina»):")
    for k, w in sorted(zip(P.MODELOS["completo"], modelos["completo"]["coef"]), key=lambda kv: -abs(kv[1])):
        print(f"  {k:24s} {w:+.2f}")
    escribir_bot(modelos)
    comprobar(modelos, H + [c for c, _ in largas[:40]])


def escribir_bot(modelos):
    reglas = open(os.path.join(GUARDIAN, "reglas.py"), encoding="utf-8").read()
    plantilla = open(os.path.join(GUARDIAN, "plantilla_bot.py"), encoding="utf-8").read()
    plantilla = plantilla.split('"""', 2)[2]  # quita el docstring de la plantilla
    plantilla = re.sub(r"^from reglas import .*\n", "", plantilla, flags=re.M)
    lineas = ["PARAMS = {"]
    for nombre, m in modelos.items():
        lineas += [f"    {nombre!r}: {{",
                   f"        \"media\": {[round(float(v), 6) for v in m['media']]!r},",
                   f"        \"desv\": {[round(float(v), 6) for v in m['desv']]!r},",
                   f"        \"coef\": {[round(float(v), 6) for v in m['coef']]!r},",
                   f"        \"sesgo\": {round(float(m['sesgo']), 6)!r},",
                   f"        \"corte_unica\": {m['corte_unica']!r},  # respuestas de una frase",
                   f"        \"corte_larga\": {m['corte_larga']!r},  # varias frases (noisy-OR)",
                   "    },"]
    params = "\n".join(lineas + ["}"])
    plantilla = re.sub(r"PARAMS = None", lambda _: params, plantilla)
    cortes = "; ".join(f"{n} {m['corte_unica']:.2f} (una frase) y {m['corte_larga']:.2f} (varias)" for n, m in modelos.items())
    cabecera = ('"""El Guardián: bot del equipo Syntax Terror (P13, P, D, J). FICHERO GENERADO, no editar a mano.\n\n'
                "Se genera con `python herramientas/montar_bot.py` (desde kit/) a partir de\n"
                "guardian/reglas.py (P), las señales de D (guardian/plantilla_bot.py) y los pesos\n"
                "aprendidos en el train (casos H x3 + otras familias). Solo usa la librería estándar.\n\n"
                "Cómo decide: señales del juez LLM y del NLI (peor oración) + reglas de verificación\n"
                "(cifras, matices, alcance, causas; y cuentas que cuadran como pista a favor)\n"
                "-> regresión logística ponderada 4:1 -> score. Dos modelos según lo que traiga el caso\n"
                "(completo / sin juez LLM; sin señales, sospechoso). En respuestas de varias frases\n"
                "se puntúa cada frase y se combinan con noisy-OR (probabilidad de que alguna mienta).\n"
                f"Cortes óptimos fuera de muestra, recolocados en 0,5: {cortes}.\n"
                '"""\n')
    reglas = reglas.split('"""', 2)[2]  # el docstring de reglas.py va en el README
    salida = os.path.join(KIT, "mi_bot.py")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(cabecera + "\n# " + "=" * 75 + "\n# REGLAS DE VERIFICACIÓN (P)  — copia de guardian/reglas.py\n# "
                + "=" * 75 + reglas + "\n\n# " + "=" * 75 + "\n# SEÑALES (D) Y MODELO\n# " + "=" * 75 + plantilla)
    print(f"\nEscrito {salida}")


def comprobar(modelos, casos):
    spec = importlib.util.spec_from_file_location("mi_bot_generado", os.path.join(KIT, "mi_bot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bot = mod.Guardian()
    P.PARAMS = {n: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()} for n, m in modelos.items()}
    muestra = casos[:60] + casos[-40:]
    esperado = np.array([P.score_caso(c) for c in muestra])
    obtenido = np.array([bot.score({k: c[k] for k in ("pregunta", "fuentes", "respuesta", "senales")}) for c in muestra])
    dif = float(np.max(np.abs(esperado - obtenido)))
    print(f"Comprobación: mi_bot.py reproduce los modelos en {len(muestra)} casos de una y varias frases "
          f"(diferencia máxima {dif:.2e})")
    if dif > 1e-4:
        raise SystemExit("ERROR: mi_bot.py no coincide con el modelo entrenado")


if __name__ == "__main__":
    main()
