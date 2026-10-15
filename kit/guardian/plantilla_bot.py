"""Plantilla de mi_bot.py. NO se entrega tal cual.

`python herramientas/montar_bot.py` (desde kit/) entrena los modelos y escribe
kit/mi_bot.py: pega dentro reglas.py (P) y esta plantilla, con los pesos
escritos como números. Así mi_bot.py es UN solo fichero, sin dependencias
(ni numpy), que es lo que se entrega.

Cómo decide el bot:
  1. Señales (D): veredicto y confianza del juez LLM y respaldo del NLI,
     quedándose siempre con la PEOR oración.
  2. Reglas (P): cifras que no cuadran, matices perdidos, alcance, causas
     añadidas y, como pista a favor, cuentas que sí cuadran.
  3. Regresión logística (pesos aprendidos en los casos H, con un escape pesando
     4 veces más que una falsa alarma) y un sesgo desplazado para que el corte
     óptimo, elegido fuera de muestra, caiga exactamente en 0,5.

Hay dos modelos según lo que traiga cada caso (pensado para el giro, por si
cambia el formato): completo y sin juez LLM. Sin LLM ni NLI, se marca como sospechoso.
"""
import math
import unicodedata

from reglas import RECOMENDADAS, reglas_caso  # en mi_bot.py esta línea se sustituye por reglas.py entero

SENALES_LLM = ["llm_no_respaldada", "llm_contradicha", "llm_confianza_duda"]
SENALES_NLI = ["nli_min_respaldo", "frag_min_respaldo"]
# Fuera: nli_max_contradiccion y nli_respaldo_todas_min. En H no distinguen (AUROC ~0,5)
# y el modelo les daba pesos con el signo al revés: era ruido.

MODELOS = {
    "completo": SENALES_LLM + SENALES_NLI + RECOMENDADAS,
    "sin_llm": SENALES_NLI + RECOMENDADAS,
}
# Sin LLM ni NLI no hay modelo: uno «solo reglas» daba −0,05 fuera de H (peor que
# marcarlo todo), así que en ese caso se marca como sospechoso (puntuación 0 asegurada).
SIN_SENALES = 0.6


def _veredicto(v):
    """'Respaldada', 'NO RESPALDADA', 'no-respaldada'... -> respaldada / no_respaldada / contradicha / None."""
    if not isinstance(v, str):
        return None
    v = "".join(c for c in unicodedata.normalize("NFD", v.strip().lower()) if unicodedata.category(c) != "Mn")
    v = v.replace("-", "_").replace(" ", "_")
    if v.startswith("contradic"):
        return "contradicha"
    if v.startswith(("no_resp", "no_sop", "sin_resp")):
        return "no_respaldada"
    if v.startswith("respald"):
        return "respaldada"
    return None


def _num(x, defecto=0.0):
    try:
        x = float(x)
        return x if x == x else defecto  # descarta NaN
    except (TypeError, ValueError):
        return defecto


def _max(lista):
    return max((_num(v) for v in (lista or [])), default=None)


def caracteristicas_senales(caso):
    """Señales de D (kit/guardian/senales.py), con estos arreglos:
    - el original devolvía dentro del bucle y solo miraba la primera oración;
    - con "confianza": null lanzaba una excepción;
    - veredictos normalizados (mayúsculas, tildes, espacios);
    - si no hay oraciones se usan los fragmentos para el NLI.
    Devuelve también qué hay disponible (hay_llm, hay_nli) para elegir modelo.
    """
    senales = caso.get("senales") or {}
    oraciones = senales.get("oraciones") or []
    fragmentos = senales.get("fragmentos") or []
    no_resp = contra = duda = 0.0
    hay_llm = False
    min_resp = None
    for o in oraciones:
        llm = o.get("llm") or {}
        ver = _veredicto(llm.get("veredicto"))
        # Sin confianza nos quedamos con el veredicto y una confianza típica: con 0,5 el
        # bot veía una «duda» enorme en cada frase y lo marcaba todo.
        conf = min(1.0, max(0.0, _num(llm.get("confianza"), 0.9)))
        if ver == "no_respaldada":
            no_resp = max(no_resp, conf)
        elif ver == "contradicha":
            contra = max(contra, conf)
        elif ver == "respaldada":
            duda = max(duda, 1.0 - conf)
        hay_llm = hay_llm or ver is not None
        m = _max(o.get("respaldo_por_fuente"))
        if m is not None:
            min_resp = m if min_resp is None else min(min_resp, m)
    frag = None
    for f in fragmentos:
        m = _max(f.get("respaldo_por_fuente"))
        if m is not None:
            frag = m if frag is None else min(frag, m)
    if min_resp is None:
        min_resp = frag
    if frag is None:
        frag = min_resp
    hay_nli = min_resp is not None
    return {"llm_no_respaldada": no_resp, "llm_contradicha": contra, "llm_confianza_duda": duda,
            "nli_min_respaldo": min_resp if hay_nli else 0.0, "frag_min_respaldo": frag if hay_nli else 0.0,
            "hay_llm": hay_llm, "hay_nli": hay_nli}


def caracteristicas(caso):
    """Todas las características del caso (señales de D + reglas de P) en un dict."""
    d = caracteristicas_senales(caso)
    d.update(reglas_caso(caso))
    return d


def elegir_modelo(d):
    if d["hay_llm"] and d["hay_nli"]:
        return "completo"
    return "sin_llm" if d["hay_nli"] else None


def vector(d, modelo):
    return [float(d[k]) for k in MODELOS[modelo]]


# === PARAMETROS (los escribe herramientas/montar_bot.py) ===
PARAMS = None
# === FIN PARAMETROS ===


class Guardian:
    def score(self, caso):
        try:
            d = caracteristicas(caso)
            modelo = elegir_modelo(d)
            if modelo is None:
                return SIN_SENALES
            p = PARAMS[modelo]
            z = p["sesgo"]
            for v, m, s, w in zip(vector(d, modelo), p["media"], p["desv"], p["coef"]):
                z += w * (v - m) / s
            prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            return min(1.0, max(0.0, prob))
        except Exception:
            # Ante cualquier fallo, mejor una falsa alarma (cuesta 1) que un escape (cuesta 4)
            return 0.6
