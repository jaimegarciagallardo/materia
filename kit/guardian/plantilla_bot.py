"""Plantilla de mi_bot.py. NO se entrega tal cual.

`python herramientas/montar_bot.py` (desde kit/) entrena el modelo y escribe
kit/mi_bot.py: pega dentro reglas.py (P) y esta plantilla, con los pesos
escritos como números. Así mi_bot.py es UN solo fichero, sin dependencias
(ni numpy), que es lo que se entrega.

Cómo decide el bot:
  1. Señales (D): veredicto y confianza del juez LLM y respaldo/contradicción del
     NLI, quedándose siempre con la PEOR oración.
  2. Reglas (P): cifras que no cuadran, matices perdidos, alcance, causas añadidas.
  3. Regresión logística (pesos aprendidos en los casos H, con un escape pesando
     4 veces más que una falsa alarma) y un sesgo desplazado para que el corte
     óptimo, elegido fuera de muestra, caiga exactamente en 0,5.
"""
import math

from reglas import RECOMENDADAS, reglas_caso  # en mi_bot.py esta línea se sustituye por reglas.py entero


def caracteristicas_senales(caso):
    """Señales de D (kit/guardian/senales.py), con dos fallos corregidos:
    - el original devolvía dentro del bucle y solo miraba la primera oración;
    - con "confianza": null lanzaba una excepción.
    Además añade llm_sin_veredicto (el juez no contestó).
    """
    oraciones = (caso.get("senales") or {}).get("oraciones") or []
    fragmentos = (caso.get("senales") or {}).get("fragmentos") or []
    if not oraciones:
        return {"llm_no_respaldada": 0.5, "llm_contradicha": 0.0, "llm_confianza_duda": 0.5,
                "llm_sin_veredicto": 1.0, "nli_min_respaldo": 0.0, "nli_max_contradiccion": 0.5,
                "nli_respaldo_todas_min": 0.0, "frag_min_respaldo": 0.0}
    no_resp = contra = duda = sin_ver = 0.0
    min_resp = min_todas = 1.0
    max_contra = 0.0
    for o in oraciones:
        llm = o.get("llm") or {}
        ver = llm.get("veredicto")
        conf = llm.get("confianza")
        conf = 0.5 if conf is None else float(conf)
        if ver == "no_respaldada":
            no_resp = max(no_resp, conf)
        elif ver == "contradicha":
            contra = max(contra, conf)
        elif ver == "respaldada":
            duda = max(duda, 1.0 - conf)
        else:
            sin_ver = 1.0
        min_resp = min(min_resp, max(o.get("respaldo_por_fuente") or [0.0]))
        min_todas = min(min_todas, float(o.get("respaldo_todas") or 0.0))
        max_contra = max(max_contra, max(o.get("contradiccion_por_fuente") or [0.0]),
                         float(o.get("contradiccion_todas") or 0.0))
    frag = min((max(f.get("respaldo_por_fuente") or [0.0]) for f in fragmentos), default=min_resp)
    return {"llm_no_respaldada": no_resp, "llm_contradicha": contra, "llm_confianza_duda": duda,
            "llm_sin_veredicto": sin_ver, "nli_min_respaldo": min_resp, "nli_max_contradiccion": max_contra,
            "nli_respaldo_todas_min": min_todas, "frag_min_respaldo": frag}


def caracteristicas(caso):
    """Vector de entrada del modelo: señales de D + reglas recomendadas de P."""
    s = caracteristicas_senales(caso)
    r = reglas_caso(caso)
    return [s[k] for k in NOMBRES_SENALES] + [r[k] for k in RECOMENDADAS]


NOMBRES_SENALES = ["llm_no_respaldada", "llm_contradicha", "llm_confianza_duda", "llm_sin_veredicto",
                   "nli_min_respaldo", "nli_max_contradiccion", "nli_respaldo_todas_min", "frag_min_respaldo"]

# === PARAMETROS (los escribe herramientas/montar_bot.py) ===
PARAMS = None
# === FIN PARAMETROS ===


class Guardian:
    def score(self, caso):
        try:
            x = caracteristicas(caso)
            z = PARAMS["sesgo"]
            for v, m, d, w in zip(x, PARAMS["media"], PARAMS["desv"], PARAMS["coef"]):
                z += w * (v - m) / d
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            return min(1.0, max(0.0, p))
        except Exception:
            # Ante cualquier fallo, mejor una falsa alarma (cuesta 1) que un escape (cuesta 4)
            return 0.6
