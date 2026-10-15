def caracteristicas_senales(caso: dict) -> dict:
    senales = caso.get("senales", {})
    oraciones = senales.get("oraciones", [])
    fragmentos = senales.get("fragmentos", [])
    
    # Si un caso viene sin oraciones procesadas, devolvemos valores neutros/prudentes
    if not oraciones:
        return {
            "llm_no_respaldada": 0.5,
            "llm_contradicha": 0.0,
            "llm_confianza_duda": 0.5,
            "nli_min_respaldo": 0.0,
            "nli_max_contradiccion": 0.5,
            "nli_respaldo_todas_min": 0.0,
            "frag_min_respaldo": 0.0,
        }

    # Acumuladores del LLM (Riesgo/Alarma)
    peor_llm_no_respaldada = 0.0
    peor_llm_contradicha = 0.0
    mayor_duda_llm_respaldada = 0.0

    # Acumuladores de NLI (Soporte/Seguridad)
    peor_respaldo_fuente = 1.0
    peor_respaldo_todas = 1.0
    mayor_contradiccion = 0.0

    for oracion in oraciones:
        # A. Señales del Juez LLM
        # Subdiccionario Juez Asignado
        llm = oracion.get("llm", {})
        # Etiqueta que Juez decide sobre la frase (3 opciones únicas)
        veredicto = llm.get("veredicto", "")
        # Entre 0 y 1. Certidumbre del veredicto del Juez. Cerca de 0 sin duda.
        confianza = float(llm.get("confianza", 0.0))

        # Clasificación del Riesgo
        if veredicto == "no_respaldada":
            peor_llm_no_respaldada = max(peor_llm_no_respaldada, confianza)
        elif veredicto == "contradicha":
            peor_llm_contradicha = max(peor_llm_contradicha, confianza)
        elif veredicto == "respaldada":
            # Si dice respaldada pero con poca seguridad, anotamos la duda (1 - confianza)
            mayor_duda_llm_respaldada = max(mayor_duda_llm_respaldada, 1.0 - confianza)

        # B. Señales del modelo NLI
        # Mejor respaldo entre fuentes individuales
        # Lista respaldo para cada fuente (probabilidad).
        respaldo_fuentes = oracion.get("respaldo_por_fuente", [0.0])
        # Mejor Documento
        mejor_fuente_individual = max(respaldo_fuentes, default=0.0)
        # Rastreamos el peor caso
        peor_respaldo_fuente = min(peor_respaldo_fuente, mejor_fuente_individual)
        
        # Respaldo leyendo todas las fuentes juntas
        # Puntuación NLI al leer todas las fuentes.
        respaldo_todas = float(oracion.get("respaldo_todas", 0.0))
        # Rastreamos el peor caso
        peor_respaldo_todas = min(peor_respaldo_todas, respaldo_todas)
        
        # Mayor contradicción (individual o conjunta)

        # Lista de contradicciones contra cada documento individual (0.0 = sin contradicción, 1.0 = choque total).
        # Usamos default [0.0] por seguridad si la clave no viniera en el JSON.
        contras = oracion.get("contradiccion_por_fuente", [0.0])
        
        # Contradicción evaluada por el NLI leyendo todas las fuentes juntas como un único bloque.
        contra_todas = float(oracion.get("contradiccion_todas", 0.0))
        
        # Detectamos el peor choque de esta frase concreta:
        # max(contras, default=0.0) busca la fuente individual con la que más choca.
        # Luego comparamos ese valor con 'contra_todas' y nos quedamos con la señal de alarma más fuerte.
        max_contra_esta_oracion = max(max(contras, default=0.0), contra_todas)
        
        # Alarma global para toda la respuesta:
        # Si al menos una frase choca frontalmente con los textos (ej. 0.92),
        # 'mayor_contradiccion' guardará ese pico máximo para delatar la alucinación.
        mayor_contradiccion = max(mayor_contradiccion, max_contra_esta_oracion)

        # Señales a nivel de fragmento (trozos más pequeños delimitados por comas o conjunciones)
        peor_respaldo_fragmento = 1.0
        if fragmentos:
            for f in fragmentos:
                f_resp = max(f.get("respaldo_por_fuente", [0.0]), default=0.0)
                peor_respaldo_fragmento = min(peor_respaldo_fragmento, f_resp)
        else:
            # Si no hay fragmentos en este caso, usamos el peor respaldo de oración como sustituto
            peor_respaldo_fragmento = peor_respaldo_fuente

        # Devolvemos el diccionario de características cuantitativas en [0.0, 1.0]
        return {
            "llm_no_respaldada": float(peor_llm_no_respaldada),
            "llm_contradicha": float(peor_llm_contradicha),
            "llm_confianza_duda": float(mayor_duda_llm_respaldada),
            "nli_min_respaldo": float(peor_respaldo_fuente),
            "nli_max_contradiccion": float(mayor_contradiccion),
            "nli_respaldo_todas_min": float(peor_respaldo_todas),
            "frag_min_respaldo": float(peor_respaldo_fragmento),
        }