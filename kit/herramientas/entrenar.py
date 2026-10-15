import json
import os
import sys
import numpy as np

# Añadimos la raíz del repositorio al path de Python para poder importar 'senales.py'
# sin importar desde qué subcarpeta ejecutemos el comando en la terminal.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from kit.guardian.senales import caracteristicas_senales


def cargar_datos_h(ruta_train="train.jsonl"):
    """
    Lee el archivo de entrenamiento y extrae únicamente los casos cuyo ID empieza por 'H'.
    
    ¿Por qué solo 'H'?
    El dataset 'train.jsonl' tiene más de 1.700 ejemplos, pero muchos son sintéticos simples.
    Los casos con prefijo 'H' (631 casos) son los únicos que reproducen la dificultad real,
    la estructura de varias oraciones y los patrones del conjunto de evaluación oficial.
    Validar o entrenar con casos no-H contaminaría los pesos con sesgos irrelevantes.
    """
    casos = []
    with open(ruta_train, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            dato = json.loads(linea)
            
            # Filtro estricto acordado por el equipo
            if dato.get("id", "").startswith("H"):
                casos.append(dato)
    return casos


def preparar_matrices(casos):
    """
    Convierte la lista de diccionarios en estructuras matriciales de NumPy (X e y).
    
    - X (matriz N x 7): Cada fila es una respuesta; cada columna es una señal numérica calculada por D.
    - y (vector N): 1.0 si la respuesta contiene al menos una alucinación; 0.0 si es 100 % fiel.
    """
    X = []
    y = []
    nombres_caracteristicas = []
    
    for caso in casos:
        # 1. Extraemos las 7 señales numéricas diseñadas en kit/guardian/senales.py
        feats = caracteristicas_senales(caso)
        
        # 2. Ordenamos alfabéticamente las claves una sola vez.
        # En Python los diccionarios pueden iterarse en distinto orden si no se fijan;
        # ordenar garantiza que la columna 0 siempre sea la misma característica en todas las filas.
        if not nombres_caracteristicas:
            nombres_caracteristicas = sorted(feats.keys())
            
        fila = [float(feats[k]) for k in nombres_caracteristicas]
        X.append(fila)
        
        # 3. Normalizamos la etiqueta real del caso a formato binario (0.0 o 1.0).
        # Dependiendo del split, la clave 'etiqueta' puede venir como texto ("alucinacion", "no_fiel")
        # o como booleano/entero.
        etiqueta_str = str(caso.get("etiqueta", "")).lower()
        es_alucinacion = 1.0 if etiqueta_str in ("alucinacion", "no_fiel", "true", "1") else 0.0
        y.append(es_alucinacion)
        
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), nombres_caracteristicas


def sigmoide(z):
    """
    Función de activación logística estándar: sigma(z) = 1 / (1 + e^(-z))
    Mapea cualquier valor real (-inf, +inf) al rango de probabilidades (0.0, 1.0).
    
    np.clip(z, -20.0, 20.0) es una salvaguarda de estabilidad numérica:
    evita que exp(-z) cause un desbordamiento por división o inf (OverflowError)
    si un producto escalar da un número excesivamente grande o pequeño.
    """
    z_clipped = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def entrenar_regresion_logistica(X, y, epochs=2000, lr=0.08, peso_positivo=1.8):
    """
    Entrena una regresión logística transparente usando descenso de gradiente estándar en NumPy puro.
    
    Parámetro clave -> 'peso_positivo=2.5':
    Como las bases del concurso castigan 4 veces más dejar escapar una alucinación (FN)
    que dar una falsa alarma (FP), ponderamos el gradiente de las muestras donde y=1.
    Así el optimizador aprende de forma asimétrica, penalizando con más severidad
    los errores cometidos sobre casos con alucinaciones reales.
    """
    n_samples, n_features = X.shape
    # Inicializamos pesos (w) y sesgo/bias (b) a cero
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    
    # Vector de ponderación por muestra: multiplica el error si el caso real es alucinación
    pesos_muestra = np.where(y == 1.0, peso_positivo, 1.0)
    
    for _ in range(epochs):
        # 1. Forward pass: combinación lineal z = X*w + b y cálculo de probabilidad
        z = np.dot(X, w) + b
        y_pred = sigmoide(z)
        
        # 2. Backward pass: cálculo del gradiente del error logístico ponderado
        error = (y_pred - y) * pesos_muestra
        dw = (1.0 / n_samples) * np.dot(X.T, error)
        db = (1.0 / n_samples) * np.sum(error)
        
        # 3. Descenso de gradiente: actualizar parámetros en dirección opuesta a la pendiente
        w -= lr * dw
        b -= lr * db
        
    return w, b


def encontrar_mejor_corte(probabilidades, y_true):
    """
    Simula la métrica exacta de evaluación del concurso a lo largo de un barrido de umbrales.
    
    Fórmula oficial:
        Puntuación = 1 - (4 * escapes + falsas_alarmas) / total_fieles
        
    Busca de forma exhaustiva qué punto de corte (threshold) maximiza la puntuación,
    encontrando el equilibrio óptimo entre el coste 4x del escape frente al coste 1x de la falsa alarma.
    """
    total_fieles = np.sum(y_true == 0.0)
    if total_fieles == 0:
        return 0.5, 0.0, {}
    
    mejor_score = -float("inf")
    mejor_corte = 0.5
    mejores_metricas = {}
    
    # Probamos 181 cortes equidistantes entre 0.05 y 0.95
    for corte in np.linspace(0.05, 0.95, 181):
        # Si la probabilidad predicha supera el corte, predecimos 1 (alucinación)
        pred_alucinacion = (probabilidades >= corte).astype(np.float64)
        
        # Escapes (Falsos Negativos): era alucinación (1) pero predijimos fiel (0) -> Castigo x4
        escapes = np.sum((y_true == 1.0) & (pred_alucinacion == 0.0))
        
        # Falsas alarmas (Falsos Positivos): era fiel (0) pero predijimos alucinación (1) -> Castigo x1
        falsas_alarmas = np.sum((y_true == 0.0) & (pred_alucinacion == 1.0))
        
        # Cálculo de la métrica oficial
        score = 1.0 - ((4.0 * escapes + falsas_alarmas) / total_fieles)
        
        # Guardamos el umbral que logre la nota más alta
        if score > mejor_score:
            mejor_score = score
            mejor_corte = corte
            mejores_metricas = {
                "escapes": int(escapes),
                "falsas_alarmas": int(falsas_alarmas),
                "score": float(score)
            }
            
    return mejor_corte, mejor_score, mejores_metricas


if __name__ == "__main__":
    print(">>> 1. Cargando casos de train con prefijo H...")
    casos_h = cargar_datos_h("train.jsonl")
    print(f"Total casos H encontrados: {len(casos_h)}")
    
    if not casos_h:
        print("Error: No se encontraron casos con prefijo H. Revisa la ruta de train.jsonl.")
        sys.exit(1)
        
    print(">>> 2. Extrayendo características con senales.py...")
    X, y, nombres = preparar_matrices(casos_h)
    
    print(">>> 3. Entrenando regresión logística con NumPy puro...")
    w, b = entrenar_regresion_logistica(X, y)
    
    probabilidades = sigmoide(np.dot(X, w) + b)
    
    print(">>> 4. Buscando corte óptimo con penalización 4x...")
    corte_opt, score_opt, metricas = encontrar_mejor_corte(probabilidades, y)
    
    # Calibración analítica hacia 0.50:
    # Desplazamos el término independiente para que corte_opt pase a ser exactamente 0.50
    logit_corte = float(np.log(corte_opt / (1.0 - corte_opt)))
    b_calibrado = float(b - logit_corte)
    
    # Comprobación de calibración:
    probs_calibradas = sigmoide(np.dot(X, w) + b_calibrado)
    pred_05 = (probs_calibradas >= 0.5).astype(np.float64)
    total_fieles = np.sum(y == 0.0)
    escapes_05 = int(np.sum((y == 1.0) & (pred_05 == 0.0)))
    fa_05 = int(np.sum((y == 0.0) & (pred_05 == 1.0)))
    score_05 = 1.0 - ((4.0 * escapes_05 + fa_05) / total_fieles)

    print("\n" + "="*50)
    print("RESULTADOS DEL ENTRENAMIENTO Y CALIBRACIÓN (D)")
    print("="*50)
    print(f"Score óptimo con corte original ({corte_opt:.2f}): {score_opt:.4f}")
    print(f"Score tras calibrar al corte 0.50:            {score_05:.4f}")
    print(f"Alucinaciones escapadas (FN): {escapes_05}")
    print(f"Falsas alarmas (FP):          {fa_05}")
    print("\n--- CÓDIGO CALIBRADO LISTO PARA J (mi_bot.py) ---")
    print(f"NOMBRES_FEATS = {nombres}")
    print(f"COEF = np.array({[round(float(val), 4) for val in w]})")
    print(f"SESGO = {round(b_calibrado, 4)}")
    print("UMBRAL = 0.5  # Calibrado formalmente")
    print("="*50)