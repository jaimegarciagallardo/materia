"""Mide cuánto aporta cada regla de reglas.py.

    python guardiand/evaluar_reglas.py               # informe por regla
    python guardiand/evaluar_reglas.py --ejemplos cifra_no_derivable --n 8

Para cada regla muestra, en los casos H, en cuántos fieles y en cuántas
alucinaciones salta, y la tasa de alucinación cuando salta. El subconjunto
importante es «LLM dice respaldada»: ahí es donde las reglas tienen que ayudar.
Un 35 % es la tasa base de ese subconjunto; una regla útil la sube claramente.

Al final compara un modelo sencillo (regresión logística en numpy, validación
cruzada agrupada por caso base, p. ej. H201) con solo señales frente a señales
+ reglas. Las señales son una versión mínima de lo que hará D, solo para medir.
"""
import argparse
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from reglas import NOMBRES, RECOMENDADAS, reglas_caso  # noqa: E402

KIT = os.path.join(AQUI, "..", "kit")


def cargar(nombre):
    with open(os.path.join(KIT, nombre), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def veredicto(caso):
    vs = [(o.get("llm") or {}).get("veredicto") for o in caso["senales"]["oraciones"]]
    return "respaldada" if all(v == "respaldada" for v in vs) else "otro"


def senales_minimas(c):
    """Stand-in de las características de D (solo para medir el aporte de las reglas)."""
    O = c["senales"]["oraciones"]
    F = c["senales"]["fragmentos"] or O
    peor, ws = None, -1.0
    for o in O:
        l = o.get("llm") or {}
        v, cf = l.get("veredicto"), l.get("confianza") or 0.5
        s = 1 - cf if v == "respaldada" else 0.5 if v is None else cf
        if s > ws:
            ws, peor = s, (v, cf, l.get("fuente"), o)
    v, cf, fi, o = peor
    rpf = o["respaldo_por_fuente"]
    citado = rpf[fi] if isinstance(fi, int) and fi < len(rpf) else 0.0
    mx = lambda l: max(l, default=0.0)
    return [v == "respaldada", v == "contradicha", v == "no_respaldada", v is None, cf, ws, citado,
            min(mx(x["respaldo_por_fuente"]) for x in O), min(x["respaldo_todas"] for x in O),
            min(mx(x["respaldo_por_fuente"]) for x in F), min(x["respaldo_todas"] for x in F)]


def puntuacion(y, p, t=0.5):
    fn = int(((y == 1) & (p < t)).sum())
    fp = int(((y == 0) & (p >= t)).sum())
    return 1 - (4 * fn + fp) / (y == 0).sum(), fn, fp


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / len(pos) / len(neg)


def entrenar_lr(X, y, peso_pos=4.0, l2=1.0, iters=2000):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.where(y == 1, peso_pos, 1.0)
    b = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ b))
        g = Z.T @ ((p - y) * w) / w.sum() + l2 * np.r_[b[:-1], 0] / len(y)
        b -= 0.5 * g
    return lambda Xn: 1 / (1 + np.exp(-(np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ b))), b


def cv_agrupada(X, y, grupos, k=5, semillas=(0, 1, 2)):
    res = []
    for s in semillas:
        ug = np.unique(grupos)
        np.random.default_rng(s).shuffle(ug)
        P = np.zeros(len(y))
        for f in np.array_split(ug, k):
            te = np.isin(grupos, f)
            modelo, _ = entrenar_lr(X[~te], y[~te])
            P[te] = modelo(X[te])
        mejor = max(puntuacion(y, P, t)[0] for t in np.unique(np.round(P, 3)))
        res.append((auroc(y, P), puntuacion(y, P)[0], mejor))
    return np.mean(res, axis=0)


def informe(train, test):
    H = [c for c in train if c["id"].startswith("H")]
    noH = [c for c in train if not c["id"].startswith("H")]
    feats = {id(c): reglas_caso(c, estricto=True) for c in train + test}

    def fila(casos, k):
        f0 = [c for c in casos if c["etiqueta"] == 0]
        f1 = [c for c in casos if c["etiqueta"] == 1]
        on0 = sum(feats[id(c)][k] >= 0.5 for c in f0)
        on1 = sum(feats[id(c)][k] >= 0.5 for c in f1)
        tasa = on1 / (on0 + on1) if on0 + on1 else float("nan")
        return f"{on0:4d}/{len(f0):<4d} {on1:4d}/{len(f1):<4d} {tasa:5.2f}"

    HR = [c for c in H if veredicto(c) == "respaldada"]
    base = sum(c["etiqueta"] for c in HR) / len(HR)
    print(f"Tasa base de alucinación en H con LLM=respaldada: {base:.2f} (n={len(HR)})")
    print(f"{'regla':22s} | {'H: fieles  aluc   tasa':24s} | {'H&LLM=resp: fieles aluc tasa':28s} | {'no-H: fieles aluc tasa':24s} | test")
    for k in NOMBRES:
        t = sum(feats[id(c)][k] >= 0.5 for c in test) / len(test)
        print(f"{k:22s} | {fila(H, k)} | {fila(HR, k)}   | {fila(noH, k)} | {t:4.2f}")
    print("(fieles/aluc = en cuántos salta la regla; tasa = % de alucinaciones cuando salta;"
          " test = % de casos del test en que salta)")

    y = np.array([c["etiqueta"] for c in H], float)
    g = np.array([c["id"].split("::")[0] for c in H])
    yN = np.array([c["etiqueta"] for c in noH], float)

    def X(casos, claves):
        s = np.array([senales_minimas(c) for c in casos], float)
        return np.c_[s, np.array([[feats[id(c)][k] for k in claves] for c in casos], float)] if claves else s

    print("\nRegresión logística 4:1. CV = validación cruzada agrupada en H (5 bloques × 3 semillas)."
          "\n«H→no-H» = entrenado en todo H y probado en el resto del train (otro formato: proxy del giro).")
    for nombre, claves in (("solo señales", []), ("señales + recomendadas", RECOMENDADAS),
                           ("señales + todas", NOMBRES)):
        a, p05, mejor = cv_agrupada(X(H, claves), y, g)
        modelo, _ = entrenar_lr(X(H, claves), y)
        P = modelo(X(noH, claves))
        print(f"  {nombre:24s} CV: AUROC {a:.3f} · @0.5 {p05:+.3f} · mejor corte {mejor:+.3f}"
              f"  |  H→no-H: AUROC {auroc(yN, P):.3f} · @0.5 {puntuacion(yN, P)[0]:+.3f}")
    _, b = entrenar_lr(X(H, RECOMENDADAS), y)
    print("\nPesos de las reglas recomendadas (modelo con todo H, estandarizados):")
    for k, w in sorted(zip(RECOMENDADAS, b[-len(RECOMENDADAS) - 1:-1]), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {w:+.2f}")


def ejemplos(train, regla, n, etiqueta):
    from reglas import caracteristicas_reglas, _fuentes_relevantes  # noqa
    vistos = 0
    for c in train:
        if not c["id"].startswith("H") or (etiqueta is not None and c["etiqueta"] != etiqueta):
            continue
        if reglas_caso(c)[regla] < 0.5:
            continue
        print(f"### {c['id']} etiqueta={c['etiqueta']} tipo={c.get('tipo')} LLM={veredicto(c)}")
        print("  R:", c["respuesta"][:300])
        o = c["senales"]["oraciones"][0]
        for f in _fuentes_relevantes(o, c["fuentes"])[:3]:
            print("  F:", f[:220])
        vistos += 1
        if vistos >= n:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ejemplos", help="nombre de una regla para ver casos en los que salta")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--etiqueta", type=int, default=None, help="0 = solo fieles (falsas alarmas), 1 = solo alucinaciones")
    a = ap.parse_args()
    tr = cargar("train.jsonl")
    if a.ejemplos:
        ejemplos(tr, a.ejemplos, a.n, a.etiqueta)
    else:
        informe(tr, cargar("test_publico.jsonl"))
