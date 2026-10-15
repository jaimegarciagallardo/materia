"""Entrena el modelo y escribe kit/mi_bot.py (un solo fichero, sin dependencias).

    cd kit
    python herramientas/montar_bot.py

Qué hace:
  1. Calcula las características (señales de D + reglas de P) de los casos H.
  2. Validación cruzada AGRUPADA por tema (H201::r1..r9 siempre juntos), 5 bloques
     × 3 semillas: cada caso se puntúa con un modelo que no lo ha visto.
  3. Elige el corte con esas puntuaciones fuera de muestra (no con las de
     entrenamiento, que saldrían infladas).
  4. Entrena el modelo final con todo H y desplaza el sesgo para que ese corte
     caiga en 0,5 (el organizador corta en 0,5).
  5. Escribe mi_bot.py = reglas.py + plantilla_bot.py + pesos, y comprueba que
     el fichero escrito da exactamente las mismas puntuaciones.
"""
import importlib.util
import json
import math
import os
import re
import sys

import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARDIAN = os.path.join(KIT, "guardian")
sys.path.insert(0, GUARDIAN)
import plantilla_bot as P  # noqa: E402
from reglas import RECOMENDADAS  # noqa: E402

PESO_ESCAPE = 4.0   # un escape pesa 4 veces más que una falsa alarma
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


def entrenar(X, y):
    media, desv = X.mean(0), X.std(0)
    desv[desv < 1e-9] = 1.0
    Z = np.c_[(X - media) / desv, np.ones(len(X))]
    w = np.where(y == 1, PESO_ESCAPE, 1.0)
    b = np.zeros(Z.shape[1])
    for _ in range(3000):
        p = 1 / (1 + np.exp(-Z @ b))
        g = Z.T @ ((p - y) * w) / w.sum() + L2 * np.r_[b[:-1], 0] / len(y)
        b -= 0.5 * g
    return {"media": media, "desv": desv, "coef": b[:-1], "sesgo": b[-1]}


def predecir(m, X):
    z = ((X - m["media"]) / m["desv"]) @ m["coef"] + m["sesgo"]
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def main():
    train = cargar("train.jsonl")
    H = [c for c in train if c["id"].startswith("H")]
    noH = [c for c in train if not c["id"].startswith("H")]
    nombres = P.NOMBRES_SENALES + RECOMENDADAS
    print(f"Características ({len(nombres)}): {', '.join(nombres)}")
    X = np.array([P.caracteristicas(c) for c in H], float)
    y = np.array([c["etiqueta"] for c in H], float)
    g = np.array([c["id"].split("::")[0] for c in H])

    # Validación cruzada agrupada: probabilidades fuera de muestra
    oof = []
    for semilla in (0, 1, 2):
        temas = np.unique(g)
        np.random.default_rng(semilla).shuffle(temas)
        Pr = np.zeros(len(y))
        for bloque in np.array_split(temas, 5):
            te = np.isin(g, bloque)
            Pr[te] = predecir(entrenar(X[~te], y[~te]), X[te])
        oof.append(Pr)
    cortes = np.round(np.arange(0.05, 0.951, 0.01), 2)
    medias = [np.mean([puntuacion(y, Pr, t)[0] for Pr in oof]) for t in cortes]
    corte = float(cortes[int(np.argmax(medias))])
    print(f"\nValidación cruzada agrupada en H ({len(H)} casos, {len(np.unique(g))} temas):")
    print(f"  AUROC {np.mean([auroc(y, Pr) for Pr in oof]):.3f}")
    print(f"  mejor corte fuera de muestra: {corte:.2f} -> puntuación {max(medias):+.3f}")
    for t in (0.3, 0.4, 0.5, 0.6):
        print(f"  (con corte {t:.1f}: {np.mean([puntuacion(y, Pr, t)[0] for Pr in oof]):+.3f})")

    # Modelo final con todo H; el sesgo se desplaza para que el corte quede en 0,5
    m = entrenar(X, y)
    m["sesgo"] -= math.log(corte / (1 - corte))
    XN = np.array([P.caracteristicas(c) for c in noH], float)
    yN = np.array([c["etiqueta"] for c in noH], float)
    pN = predecir(m, XN)
    s, fn, fp = puntuacion(yN, pN, 0.5)
    print(f"\nControl en no-H (otro formato, no usado para entrenar): puntuación@0,5 {s:+.3f} · AUROC {auroc(yN, pN):.3f}"
          f" · {fn} escapes · {fp} falsas alarmas")

    print("\nPesos (estandarizados; positivo = empuja a «alucina»):")
    for k, w in sorted(zip(nombres, m["coef"]), key=lambda kv: -abs(kv[1])):
        print(f"  {k:24s} {w:+.2f}")

    escribir_bot(m, nombres, corte)
    comprobar(m, H)


def escribir_bot(m, nombres, corte):
    reglas = open(os.path.join(GUARDIAN, "reglas.py"), encoding="utf-8").read()
    plantilla = open(os.path.join(GUARDIAN, "plantilla_bot.py"), encoding="utf-8").read()
    plantilla = plantilla.split('"""', 2)[2]  # quita el docstring de la plantilla
    plantilla = re.sub(r"^from reglas import .*\n", "", plantilla, flags=re.M)
    params = ("PARAMS = {\n"
              f"    \"nombres\": {nombres!r},\n"
              f"    \"media\": {[round(float(v), 6) for v in m['media']]!r},\n"
              f"    \"desv\": {[round(float(v), 6) for v in m['desv']]!r},\n"
              f"    \"coef\": {[round(float(v), 6) for v in m['coef']]!r},\n"
              f"    \"sesgo\": {round(float(m['sesgo']), 6)!r},  # ya desplazado: corte {corte:.2f} -> 0,5\n"
              "}")
    plantilla = re.sub(r"PARAMS = None", lambda _: params, plantilla)
    cabecera = ('"""El Guardián: bot del equipo (P13, P, D, J). FICHERO GENERADO, no editar a mano.\n\n'
                "Se genera con `python herramientas/montar_bot.py` (desde kit/) a partir de\n"
                "guardian/reglas.py (P), las señales de D (guardian/plantilla_bot.py) y los pesos\n"
                "aprendidos en los casos H. Solo usa la librería estándar.\n\n"
                "Cómo decide: señales del juez LLM y del NLI (peor oración) + reglas de verificación\n"
                "(cifras, matices, alcance, causas) -> regresión logística ponderada 4:1 -> score.\n"
                f"El corte óptimo fuera de muestra ({corte:.2f}) se ha desplazado a 0,5.\n"
                '"""\n')
    reglas = reglas.split('"""', 2)[2]  # el docstring de reglas.py va en el README
    salida = os.path.join(KIT, "mi_bot.py")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(cabecera + "\n# " + "=" * 75 + "\n# REGLAS DE VERIFICACIÓN (P)  — copia de guardian/reglas.py\n# "
                + "=" * 75 + reglas + "\n\n# " + "=" * 75 + "\n# SEÑALES (D) Y MODELO\n# " + "=" * 75 + plantilla)
    print(f"\nEscrito {salida}")


def comprobar(m, H):
    spec = importlib.util.spec_from_file_location("mi_bot_generado", os.path.join(KIT, "mi_bot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bot = mod.Guardian()
    X = np.array([P.caracteristicas(c) for c in H[:60]], float)
    esperado = predecir(m, X)
    obtenido = np.array([bot.score({k: c[k] for k in ("pregunta", "fuentes", "respuesta", "senales")}) for c in H[:60]])
    dif = float(np.max(np.abs(esperado - obtenido)))
    print(f"Comprobación: mi_bot.py reproduce el modelo (diferencia máxima {dif:.2e})")
    if dif > 1e-4:
        raise SystemExit("ERROR: mi_bot.py no coincide con el modelo entrenado")


if __name__ == "__main__":
    main()
