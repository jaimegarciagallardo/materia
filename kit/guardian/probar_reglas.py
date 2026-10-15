"""Pruebas rápidas de reglas.py con frases inventadas (no salen del train ni del test).

    python guardiand/probar_reglas.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reglas import caracteristicas_reglas, reglas_caso  # noqa: E402

CASOS = [
    # (regla esperada, valor esperado, oración, fuentes)
    ("cifra_no_derivable", 1, "El fármaco redujo la presión un 21 %.",
     ["El fármaco redujo la presión un 12 %."]),
    ("cifra_no_derivable", 0, "Entre las dos sedes suman 1.250 socios.",
     ["La sede norte tiene 800 socios.", "La sede sur tiene 450 socios."]),
    ("cifra_no_derivable", 0, "El 25 % de los 4.000 euros, es decir, 1.000 euros, lo pagó la empresa.",
     ["El proyecto costó 4.000 euros; la empresa pagó el 25 %."]),
    ("cifra_no_derivable", 1, "La plantilla pasó de 200 a 260 personas, un 40 % más.",
     ["La plantilla pasó de 200 a 260 personas."]),
    ("cifra_no_derivable", 0, "Seis de cada diez usuarios van en autobús.",
     ["El 40 % de los usuarios va en coche; el resto, en autobús."]),
    ("ratio_falso", 1, "Sus 280 plazas triplican las 120 del centro viejo.",
     ["El centro nuevo tiene 280 plazas.", "El centro viejo tenía 120 plazas."]),
    ("ratio_falso", 0, "Sus 360 plazas triplican las 120 del centro viejo.",
     ["El centro nuevo tiene 360 plazas.", "El centro viejo tenía 120 plazas."]),
    ("modal_perdido", 1, "La obra costará 3 millones.",
     ["El ayuntamiento estima que la obra podría costar 3 millones."]),
    ("modal_perdido", 0, "La obra podría costar 3 millones.",
     ["El ayuntamiento estima que la obra podría costar 3 millones."]),
    ("modal_perdido", 1, "La ampliación permitirá atender a 500 pacientes.",
     ["La ampliación permitiría atender a 500 pacientes."]),
    ("condicion_perdida", 1, "Los jubilados viajarán gratis en el autobús.",
     ["Los jubilados viajarán gratis en el autobús siempre que estén empadronados."]),
    ("condicion_perdida", 0, "Los jubilados viajarán gratis en el autobús si están empadronados.",
     ["Los jubilados viajarán gratis en el autobús siempre que estén empadronados."]),
    ("restriccion_perdida", 1, "La agenda separada se aplica en los juzgados de la ciudad.",
     ["La agenda separada se aplica únicamente en el juzgado número 3 de la ciudad."]),
    ("intencion_como_hecho", 1, "El pleno subirá la tasa a 40 euros.",
     ["La concejala propuso subir la tasa a 40 euros."]),
    ("aproximacion_perdida", 1, "La feria recibió 150.000 visitantes.",
     ["La feria recibió más de 150.000 visitantes."]),
    ("parcial_perdido", 1, "Los comedores de la comarca sirven menú ecológico.",
     ["14 de los 23 comedores de la comarca sirven menú ecológico."]),
    ("modificador_omitido", 1, "El arancel al vino europeo bajará al 10 %.",
     ["El arancel al vino embotellado europeo bajará al 10 %."]),
    ("causa_anadida", 1, "La biblioteca, que abrió en 1987 gracias a una donación privada, tiene 64.000 volúmenes.",
     ["La biblioteca municipal abrió en 1987.", "Tiene un fondo de 64.000 volúmenes."]),
    ("causa_anadida", 0, "Con 64.000 volúmenes en su fondo, la biblioteca prestó 21.000 libros el año pasado.",
     ["Tiene un fondo de 64.000 volúmenes.", "El año pasado registró 21.000 préstamos."]),
    ("fecha_nueva", 1, "El plan arrancó en 2019.",
     ["El plan arrancó en 2021."]),
]


def main():
    fallos = 0
    for regla, esperado, oracion, fuentes in CASOS:
        r = caracteristicas_reglas(oracion, fuentes, fuentes)
        ok = (r[regla] >= 0.5) == bool(esperado)
        fallos += not ok
        print(f"{'ok   ' if ok else 'FALLO'} {regla:22s} esperado={esperado} obtenido={r[regla]:.0f} · {oracion[:60]}")
    # casos raros: nunca debe lanzar excepción
    raros = [{}, {"fuentes": [], "respuesta": ""}, {"fuentes": ["a"], "respuesta": "b", "senales": {}},
             {"fuentes": ["Texto 1."], "respuesta": "Algo.", "senales": {"oraciones": [{"texto": "Algo.", "llm": None}]}},
             {"fuentes": ["x"], "respuesta": "y", "senales": {"oraciones": [{"texto": "y", "llm": {"fuente": 7},
                                                                             "respaldo_por_fuente": []}]}}]
    for caso in raros:
        reglas_caso(caso, estricto=True)
    print(f"\n{len(CASOS) - fallos}/{len(CASOS)} pruebas correctas · {len(raros)} casos raros sin errores")
    return fallos


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
