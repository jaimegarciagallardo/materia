<<<<<<< HEAD
import re

class PreparadorDeTexto:
    """
    Se encarga del paquete P13: preprocesar los textos de las fuentes 
    y la respuesta aislando cifras y consolidando señales.
    """
    def __init__(self):
        # Expresión regular para atrapar números enteros y decimales. 
        # Crucial para evitar que el NLI nos engañe con falsas alarmas matemáticas.
        self.patron_numero = re.compile(r'\b\d+(?:\.\d+)?(?:,\d+)?\b')
    
    def extraer_cifras(self, texto: str) -> list:
        """Extrae todas las cifras de un string, ignorando los puntos de millar."""
        texto_limpio = texto.replace('.', '')
        return self.patron_numero.findall(texto_limpio)

    def procesar_caso(self, caso: dict) -> dict:
        """
        Recibe el caso crudo y devuelve una estructura lista para ser cruzada
        por el algoritmo de evaluación.
        """
        # 1. Preparación de las fuentes documentales
        fuentes_procesadas = []
        for idx, fuente in enumerate(caso.get("fuentes", [])):
            fuentes_procesadas.append({
                "indice": idx,
                "texto": fuente,
                "cifras_mencionadas": self.extraer_cifras(fuente)
            })
            
        # 2. Preparación de las oraciones de la IA
        oraciones_procesadas = []
        for oracion in caso["senales"]["oraciones"]:
            texto_or = oracion["texto"]
            
            # Consolidamos el texto y las señales en una sola estructura limpia
            oraciones_procesadas.append({
                "texto": texto_or,
                "cifras_mencionadas": self.extraer_cifras(texto_or),
                "respaldo_max_nli": max(oracion.get("respaldo_por_fuente", [0.0]), default=0.0),
                "contradiccion_max_nli": max(oracion.get("contradiccion_por_fuente", [0.0]), default=0.0),
                "respaldo_todas_nli": oracion.get("respaldo_todas", 0.0),
                "contradiccion_todas_nli": oracion.get("contradiccion_todas", 0.0),
                "veredicto_llm": oracion.get("llm", {}).get("veredicto") if oracion.get("llm") else None,
                "confianza_llm": oracion.get("llm", {}).get("confianza", 0.0) if oracion.get("llm") else 0.0
            })
            
        return {
            "fuentes": fuentes_procesadas,
            "oraciones": oraciones_procesadas
        }

class Guardian:
    def __init__(self):
        self.preparador = PreparadorDeTexto()

    def score(self, caso: dict) -> float:
        # Ejecutamos la preparación del texto (P13)
        datos = self.preparador.procesar_caso(caso)
        
        probabilidad_alucinacion = 0.0
        
        for oracion in datos["oraciones"]:
            riesgo = 0.0
            
            # Evaluamos la oración apoyándonos en los datos procesados.
            # Ejemplo: Desconfiar del NLI si hay un choque fuerte con el LLM
            if oracion["contradiccion_todas_nli"] > 0.8 and oracion["veredicto_llm"] == "respaldada":
                # Aquí podrías comprobar si las 'cifras_mencionadas' en la oración 
                # son resultado de una suma o conversión de unidades presentes en las 'fuentes'.
                riesgo = 1.0 - oracion["confianza_llm"]
            else:
                riesgo = 1.0 - oracion["respaldo_max_nli"]
                
            # Nos quedamos siempre con la oración que levante mayores sospechas
            probabilidad_alucinacion = max(probabilidad_alucinacion, riesgo)
            
=======
import re

class PreparadorDeTexto:
    """
    Se encarga del paquete P13: preprocesar los textos de las fuentes 
    y la respuesta aislando cifras y consolidando señales.
    """
    def __init__(self):
        # Expresión regular para atrapar números enteros y decimales. 
        # Crucial para evitar que el NLI nos engañe con falsas alarmas matemáticas.
        self.patron_numero = re.compile(r'\b\d+(?:\.\d+)?(?:,\d+)?\b')
    
    def extraer_cifras(self, texto: str) -> list:
        """Extrae todas las cifras de un string, ignorando los puntos de millar."""
        texto_limpio = texto.replace('.', '')
        return self.patron_numero.findall(texto_limpio)

    def procesar_caso(self, caso: dict) -> dict:
        """
        Recibe el caso crudo y devuelve una estructura lista para ser cruzada
        por el algoritmo de evaluación.
        """
        # 1. Preparación de las fuentes documentales
        fuentes_procesadas = []
        for idx, fuente in enumerate(caso.get("fuentes", [])):
            fuentes_procesadas.append({
                "indice": idx,
                "texto": fuente,
                "cifras_mencionadas": self.extraer_cifras(fuente)
            })
            
        # 2. Preparación de las oraciones de la IA
        oraciones_procesadas = []
        for oracion in caso["senales"]["oraciones"]:
            texto_or = oracion["texto"]
            
            # Consolidamos el texto y las señales en una sola estructura limpia
            oraciones_procesadas.append({
                "texto": texto_or,
                "cifras_mencionadas": self.extraer_cifras(texto_or),
                "respaldo_max_nli": max(oracion.get("respaldo_por_fuente", [0.0]), default=0.0),
                "contradiccion_max_nli": max(oracion.get("contradiccion_por_fuente", [0.0]), default=0.0),
                "respaldo_todas_nli": oracion.get("respaldo_todas", 0.0),
                "contradiccion_todas_nli": oracion.get("contradiccion_todas", 0.0),
                "veredicto_llm": oracion.get("llm", {}).get("veredicto") if oracion.get("llm") else None,
                "confianza_llm": oracion.get("llm", {}).get("confianza", 0.0) if oracion.get("llm") else 0.0
            })
            
        return {
            "fuentes": fuentes_procesadas,
            "oraciones": oraciones_procesadas
        }

class Guardian:
    def __init__(self):
        self.preparador = PreparadorDeTexto()

    def score(self, caso: dict) -> float:
        # Ejecutamos la preparación del texto (P13)
        datos = self.preparador.procesar_caso(caso)
        
        probabilidad_alucinacion = 0.0
        
        for oracion in datos["oraciones"]:
            riesgo = 0.0
            
            # Evaluamos la oración apoyándonos en los datos procesados.
            # Ejemplo: Desconfiar del NLI si hay un choque fuerte con el LLM
            if oracion["contradiccion_todas_nli"] > 0.8 and oracion["veredicto_llm"] == "respaldada":
                # Aquí podrías comprobar si las 'cifras_mencionadas' en la oración 
                # son resultado de una suma o conversión de unidades presentes en las 'fuentes'.
                riesgo = 1.0 - oracion["confianza_llm"]
            else:
                riesgo = 1.0 - oracion["respaldo_max_nli"]
                
            # Nos quedamos siempre con la oración que levante mayores sospechas
            probabilidad_alucinacion = max(probabilidad_alucinacion, riesgo)
            
>>>>>>> ca531c25368da35e58dd8d77154659475da08666
        return probabilidad_alucinacion