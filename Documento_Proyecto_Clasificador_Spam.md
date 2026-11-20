# Proyecto: Clasificador de Textos Spam
## Recuperación de Información mediante Aprendizaje Supervisado

---

## 1. Introducción

### 1.1 Planteamiento del Problema

El spam representa uno de los desafíos más significativos en la comunicación digital moderna. Más allá de ser una simple molestia, el spam constituye un problema multidimensional que afecta la productividad, consume recursos de tiempo y representa un vector potencial para la distribución de malware y ataques de phishing.

La saturación de bandejas de entrada con mensajes no solicitados genera:
- **Pérdida de tiempo**: Los usuarios dedican minutos valiosos diariamente a identificar y eliminar spam
- **Riesgos de seguridad**: Mensajes maliciosos que pueden contener virus, ransomware o intentos de robo de información
- **Sobrecarga de información**: Mensajes legítimos pueden perderse entre el spam

### 1.2 Objetivo del Proyecto

El objetivo principal de este proyecto es desarrollar un sistema automatizado de detección de spam mediante técnicas de Aprendizaje Supervisado, específicamente implementando un clasificador binario que distinga efectivamente entre:
- **Ham**: Mensajes legítimos y deseados
- **Spam**: Mensajes no solicitados o maliciosos

Aunque esta tarea se enmarca dentro del campo de "Recuperación de Información", se aborda mediante algoritmos de clasificación supervisada, buscando construir un sistema robusto con alta capacidad predictiva y mínima tasa de error.

---

## 2. Estado del Arte

### 2.1 Algoritmos Clásicos de Aprendizaje Supervisado

Para este proyecto se implementaron tres algoritmos fundamentales del aprendizaje automático, cada uno con características distintivas que los hacen adecuados para la clasificación de texto:

#### 2.1.1 Naive Bayes (NB)

**Fundamento Teórico:**
Naive Bayes es un clasificador probabilístico basado en el teorema de Bayes con la asunción de independencia condicional entre características.

**Ventajas:**
- **Rapidez computacional**: Entrenamiento y predicción extremadamente rápidos
- **Eficiencia con alta dimensionalidad**: Maneja bien grandes vocabularios
- **Simplicidad de implementación**: Requiere pocos parámetros de configuración
- **Efectividad en texto**: La asunción de independencia funciona sorprendentemente bien en clasificación de documentos

**Aplicabilidad al problema:**
En el contexto del spam, donde las palabras pueden considerarse relativamente independientes, Naive Bayes ofrece un balance óptimo entre simplicidad y efectividad.

#### 2.1.2 k-Nearest Neighbours (kNN)

**Fundamento Teórico:**
kNN es un algoritmo de aprendizaje basado en instancias que clasifica nuevas observaciones según la "cercanía" con ejemplos de entrenamiento.

**Principio de funcionamiento:**
1. Calcula la distancia del nuevo mensaje a todos los mensajes de entrenamiento
2. Selecciona los k vecinos más cercanos
3. Asigna la clase por mayoría de votos

**Ventajas:**
- **No paramétrico**: No asume distribución específica de los datos
- **Adaptabilidad**: Se ajusta automáticamente a patrones complejos
- **Intuitivo**: Su lógica es fácil de entender y explicar

**Desafíos:**
- Requiere una métrica de distancia apropiada para vectores de texto
- Puede ser computacionalmente costoso con grandes conjuntos de datos
- Sensible a características irrelevantes

#### 2.1.3 Support Vector Machines (SVM)

**Fundamento Teórico:**
Las Máquinas de Vectores de Soporte buscan el hiperplano óptimo que maximiza el margen de separación entre clases.

**Características distintivas:**
- **Maximización de margen**: Encuentra la frontera de decisión más robusta
- **Kernel trick**: Puede manejar separaciones no lineales mediante transformaciones
- **Efectividad en alta dimensionalidad**: Excelente rendimiento con vectores TF-IDF de alta dimensión

**Ventajas para clasificación de spam:**
- Alta capacidad de generalización
- Resistente al sobreajuste en espacios de alta dimensión
- Históricamente ha demostrado excelentes resultados en clasificación de texto

### 2.2 Justificación de la Comparación

La implementación comparativa de estos tres algoritmos permite:
1. Identificar cuál ofrece mejor rendimiento en el contexto específico del spam
2. Comprender las fortalezas y debilidades de cada aproximación
3. Validar la hipótesis de que SVM es superior para este tipo de tarea
4. Establecer un benchmark para futuras mejoras

---

## 3. Metodología

### 3.1 Pipeline de Procesamiento

El desarrollo del clasificador sigue un pipeline estructurado en cinco etapas:

#### **Fase 0: Preprocesamiento y Limpieza del Texto**

Esta etapa es crucial para la calidad de los resultados finales. El texto crudo contiene ruido que puede afectar negativamente el rendimiento del modelo.

**Pasos implementados:**

1. **Normalización a minúsculas**
   - Objetivo: Eliminar distinción entre mayúsculas y minúsculas
   - Ejemplo: "SPAM" y "spam" se tratan como la misma palabra

2. **Eliminación de URLs**
   - Rationale: Las URLs son altamente variables pero poco informativas
   - Método: Expresiones regulares para detectar patrones http/https

3. **Eliminación de números**
   - Justificación: Los valores numéricos específicos rara vez aportan valor semántico
   - Excepción: Se podrían preservar en versiones futuras si se detecta relevancia

4. **Remoción de puntuación**
   - Objetivo: Reducir vocabulario sin pérdida de información semántica
   - Caracteres eliminados: .,!?;:"'()[]{}

5. **Tokenización**
   - Proceso: División del texto en palabras individuales (tokens)
   - Base para análisis posterior

6. **Eliminación de Stop Words**
   - Stop words: Palabras funcionales de alta frecuencia ("el", "la", "de", "en")
   - Justificación: Aportan poco valor discriminativo entre spam y ham
   - Conjunto utilizado: Stopwords en inglés de NLTK (198 palabras)

7. **Stemming (Lematización)**
   - Algoritmo: Porter Stemmer
   - Propósito: Reducir palabras a su raíz morfológica
   - Ejemplo: "running", "runs", "ran" → "run"
   - Ventaja: Reduce dimensionalidad y captura conceptos equivalentes

**Impacto del preprocesamiento:**
Esta fase reduce significativamente la dimensionalidad del espacio de características mientras preserva la información semántica relevante, resultando en modelos más eficientes y precisos.

#### **Fase 1: División Train/Test**

**División de datos:**
- **Conjunto de entrenamiento**: 70% de los datos (3,545 mensajes)
- **Conjunto de prueba**: 30% de los datos (1,520 mensajes)
- **Método**: División estratificada (`stratify=y`), de modo que ambos conjuntos conservan la misma proporción de spam (11.5%)
- **Random seed**: 42 (para reproducibilidad)

La división se realiza **antes** de la vectorización. Esto es importante: si TF-IDF se ajusta con todo el dataset, el vocabulario y los pesos IDF incorporan información del conjunto de prueba y la evaluación deja de ser imparcial (*data leakage*). Una versión anterior del proyecto cometía este error.

#### **Fase 2: Vectorización TF-IDF**

Los modelos de aprendizaje automático requieren representaciones numéricas. TF-IDF (Term Frequency-Inverse Document Frequency) es el método estándar para convertir texto en vectores.

**Componentes de TF-IDF:**

1. **TF (Term Frequency)**
   - Mide la frecuencia de un término en un documento
   - Se usa la versión sublineal: TF(t,d) = 1 + log(frecuencia de t en d), para que una palabra repetida muchas veces no domine el vector

2. **IDF (Inverse Document Frequency)**
   - Penaliza términos que aparecen en muchos documentos
   - Formula (versión suavizada de scikit-learn): IDF(t) = log((1 + N) / (1 + df(t))) + 1
     - N = número total de documentos
     - df(t) = número de documentos que contienen el término t

3. **TF-IDF Score**
   - Combinación: TF-IDF(t,d) = TF(t,d) × IDF(t), y cada vector se normaliza a longitud 1
   - Interpretación: Un término tiene alta importancia si aparece frecuentemente en un documento específico pero raramente en el corpus general

**Configuración utilizada:**
- `ngram_range=(1, 2)`: se usan palabras individuales y **pares de palabras** (bigramas), como "free entri" o "call now", que capturan expresiones típicas del spam
- `sublinear_tf=True`: TF logarítmico
- `min_df=2`: se ignoran términos que aparecen en un solo mensaje (ruido, errores de escritura)
- `dtype=float32`: reduce a la mitad la memoria de la matriz

**Ventajas de TF-IDF:**
- Captura la importancia contextual de las palabras
- Reduce el peso de palabras comunes
- Genera representaciones dispersas (sparse) eficientes en memoria
- Estándar en la industria para clasificación de texto

**Resultado:**
Matriz de características de dimensión (n_documentos × n_vocabulario) donde cada documento es representado como un vector numérico. Con el conjunto de entrenamiento se obtiene un vocabulario de 4,799 términos (unigramas y bigramas).

#### **Fase 3: Entrenamiento**

Cada modelo se implementa como un `Pipeline` de scikit-learn (TF-IDF → clasificador), lo que garantiza que la vectorización se ajuste únicamente con los datos de entrenamiento, también dentro de la validación cruzada.

**Proceso de entrenamiento:**
1. Los tres modelos (NB, kNN, SVM) se entrenan **independientemente** con los mismos datos de entrenamiento
2. Configuraciones utilizadas:
   - **Naive Bayes**: MultinomialNB con suavizado `alpha=0.1`
   - **kNN**: k=5 vecinos, **distancia coseno** y voto ponderado por distancia
   - **SVM**: LinearSVC con C=1.0

3. Cada modelo aprende a distinguir patrones spam vs ham basándose en las representaciones TF-IDF

**Elección de la distancia en kNN:** la distancia euclidiana (la predeterminada) se ve afectada por la longitud de los mensajes y por la alta dimensionalidad. La distancia coseno compara la *orientación* de los vectores, es decir, qué palabras contienen y en qué proporción, y es la medida estándar para comparar documentos en Recuperación de Información.

#### **Fase 4: Evaluación**

1. Evaluación en el conjunto de prueba (1,520 mensajes) con precision, recall, F1 y matriz de confusión
2. **Validación cruzada estratificada de 5 particiones** sobre todo el dataset, para medir la variabilidad de los resultados y no depender de una sola división

### 3.2 Conjunto de Datos

**Características del dataset utilizado:**
- **Fuente**: SMS Spam Collection Dataset
- **Tamaño total**: 5,572 mensajes SMS
- **Mensajes eliminados**: 507
  - 6 mensajes quedan vacíos tras la limpieza
  - 501 mensajes son **duplicados** (idénticos tras la limpieza)
- **Mensajes utilizados**: 5,065
- **Distribución de clases (tras eliminar duplicados)**:
  - Ham: 4,481 mensajes (88.5%)
  - Spam: 584 mensajes (11.5%)

**¿Por qué eliminar duplicados?**
El dataset contiene muchos mensajes repetidos, sobre todo promociones de spam enviadas a varios destinatarios: de los 747 mensajes de spam originales, 163 son repeticiones. Si un mensaje repetido cae a la vez en entrenamiento y en prueba, el modelo se evalúa con un ejemplo que ya vio, lo que infla las métricas. Eliminarlos da una estimación más honesta de cómo se comportaría el sistema con mensajes nuevos.

**Análisis del desequilibrio:**
El dataset presenta un desequilibrio moderado hacia la clase ham (ratio ~7.7:1). Aunque no ideal, este desequilibrio refleja la distribución real del problema, donde el spam suele ser menos frecuente que los mensajes legítimos. Por esta razón, el modelo ganador se elige por el **F1 de la clase spam** y no por el accuracy: un clasificador que respondiera "ham" a todo tendría 88.5% de accuracy sin detectar un solo spam.

**Calidad de las etiquetas:**
- Dataset pre-etiquetado manualmente
- Sin necesidad de generación artificial de datos

**Características textuales:**
- Longitud variable (desde mensajes de una palabra hasta párrafos)
- Lenguaje: Inglés informal (SMS)
- Vocabulario único tras limpieza: 7,103 palabras
- Promedio de 8.9 palabras por mensaje tras la limpieza

### 3.3 Métricas de Evaluación

Para una evaluación exhaustiva del rendimiento, se utilizan múltiples métricas complementarias:

#### 3.3.1 Precisión (Precision)

**Definición:**
Proporción de mensajes clasificados como spam que realmente son spam.

**Formula:**
```
Precision = VP / (VP + FP)
```
Donde:
- VP = Verdaderos Positivos (spam correctamente clasificado)
- FP = Falsos Positivos (ham incorrectamente clasificado como spam)

**Importancia:**
- Métrica crítica para evitar que mensajes legítimos sean marcados como spam
- Falsos positivos pueden causar pérdida de información importante
- En contextos empresariales, un falso positivo puede ser más costoso que un falso negativo

#### 3.3.2 Exhaustividad / Recall

**Definición:**
Proporción del spam real que fue correctamente identificado.

**Formula:**
```
Recall = VP / (VP + FN)
```
Donde:
- FN = Falsos Negativos (spam incorrectamente clasificado como ham)

**Importancia:**
- Mide la capacidad del modelo para detectar todo el spam existente
- Alto recall significa que poco spam pasa desapercibido
- Crítico para protección contra amenazas de seguridad

#### 3.3.3 F1-Score

**Definición:**
Media armónica entre Precision y Recall.

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Ventajas:**
- Balance óptimo entre Precision y Recall
- Útil cuando se busca equilibrio entre ambas métricas
- Penaliza desequilibrios extremos

#### 3.3.4 Accuracy (Exactitud)

**Definición:**
Proporción total de predicciones correctas.

**Formula:**
```
Accuracy = (VP + VN) / (VP + VN + FP + FN)
```

**Consideraciones:**
- Métrica general del rendimiento
- Puede ser engañosa con datos desbalanceados
- Complementa otras métricas más específicas

#### 3.3.5 Matriz de Confusión

**Estructura:**
```
                Predicho Ham    Predicho Spam
Real Ham            VN              FP
Real Spam           FN              VP
```

**Utilidad:**
- Visualización directa de errores
- Identifica patrones de confusión
- Permite análisis detallado de tipos de error

---

## 4. Resultados Obtenidos

### 4.1 Rendimiento General de los Modelos

Tras el entrenamiento y evaluación en el conjunto de prueba (1,520 mensajes: 1,345 ham y 175 spam), se obtuvieron los siguientes resultados:

#### **Ranking por F1 de spam (conjunto de prueba):**

| Posición | Modelo | Accuracy | F1 spam |
|---|---|---|---|
| 🥇 | **Naive Bayes** | 97.96% | 0.9086 |
| 🥈 | **SVM** | 97.83% | 0.9032 |
| 🥉 | **kNN** | 96.97% | 0.8544 |

#### **Validación cruzada estratificada (5 particiones):**

| Modelo | Accuracy (media ± desv.) | F1 macro (media ± desv.) |
|---|---|---|
| Naive Bayes | 98.05% ± 0.48 | 0.9494 ± 0.0135 |
| SVM | 97.95% ± 0.43 | 0.9467 ± 0.0121 |
| kNN | 97.04% ± 0.57 | 0.9187 ± 0.0180 |

La diferencia entre Naive Bayes y SVM (0.1 puntos de accuracy) es mucho menor que la variación entre particiones (~0.45 puntos), por lo que **ambos modelos deben considerarse empatados**. kNN queda claramente por debajo.

### 4.2 Análisis Detallado por Modelo

#### 4.2.1 Naive Bayes - Primer Lugar (empatado con SVM)

**Métricas globales:**
- **Accuracy**: 97.96%
- **Rendimiento**: Excelente, especialmente considerando su simplicidad

**Desempeño por clase:**

**Clase Ham:**
- Precision: 98.45%
- Recall: 99.26%
- F1-Score: 98.85%

**Clase Spam:**
- Precision: 93.90%
- Recall: 88.00%
- F1-Score: 90.86%

**Matriz de Confusión:**
```
                Predicho Ham    Predicho Spam
Real Ham          1,335            10
Real Spam            21           154
```

**Análisis de errores:**
- 10 falsos positivos (ham clasificado como spam)
- 21 falsos negativos (spam no detectado)

**Fortalezas:**
- Mejor F1 de spam y mejor accuracy media en validación cruzada
- Muy rápido en entrenamiento y predicción
- Fue el único modelo que clasificó correctamente los 5 mensajes nuevos de prueba (sección 4.4)

**Conclusión Naive Bayes:**
Con el suavizado ajustado (`alpha=0.1`) y bigramas, Naive Bayes alcanza el rendimiento de SVM. Su simplicidad y rapidez lo hacen ideal para aplicaciones en tiempo real con volumen alto de mensajes.

#### 4.2.2 Support Vector Machine (SVM) - Primer Lugar (empatado con Naive Bayes)

**Métricas globales:**
- **Accuracy**: 97.83%
- **Rendimiento general**: Excelente en ambas clases

**Desempeño por clase:**

**Clase Ham:**
- Precision: 98.45%
- Recall: 99.11%
- F1-Score: 98.78%

**Clase Spam:**
- Precision: 92.77%
- Recall: 88.00%
- F1-Score: 90.32%

**Matriz de Confusión:**
```
                Predicho Ham    Predicho Spam
Real Ham          1,333            12
Real Spam            21           154
```

**Análisis de errores:**
- 12 falsos positivos (ham clasificado como spam)
- 21 falsos negativos (spam que pasó como ham)
- Detecta exactamente el mismo número de spam que Naive Bayes (154 de 175)

**Conclusión SVM:**
SVM es el modelo más estable (menor desviación estándar en validación cruzada) y obtiene el mejor resultado cuando se conservan los duplicados (sección 4.6), pero sobre datos sin duplicados no supera a Naive Bayes.

#### 4.2.3 k-Nearest Neighbours (kNN) - Tercer Lugar

**Métricas globales:**
- **Accuracy**: 96.97%
- **Rendimiento**: Bueno, pero inferior a los otros modelos

**Desempeño por clase:**

**Clase Ham:**
- Precision: 97.10%
- Recall: 99.55%
- F1-Score: 98.31%

**Clase Spam:**
- Precision: 95.74%
- Recall: 77.14%
- F1-Score: 85.44%

**Matriz de Confusión:**
```
                Predicho Ham    Predicho Spam
Real Ham          1,339             6
Real Spam            40           135
```

**Análisis de errores:**
- Solo 6 falsos positivos (el menor de los tres modelos)
- 40 falsos negativos (23% del spam no detectado)

**Fortalezas:**
- La mayor precision de spam: casi nunca marca como spam un mensaje legítimo
- Mejoró drásticamente con la distancia coseno: en la versión anterior (distancia euclidiana) solo detectaba el 32% del spam

**Debilidades:**
- Recall de spam más bajo que los otros modelos (77%)
- Es el más lento en predicción, porque compara cada mensaje con todos los de entrenamiento

**Conclusión kNN:**
Con una métrica de distancia adecuada para texto, kNN se vuelve competitivo, aunque sigue dejando pasar más spam que Naive Bayes y SVM.

### 4.3 Comparación Visual

```
Accuracy en validación cruzada (media de 5 particiones):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Naive Bayes   ████████████████████ 98.05%
SVM           ████████████████████ 97.95%
kNN           ███████████████████  97.04%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

El notebook `clasificador_spam.ipynb` incluye la gráfica de barras con las desviaciones estándar y las matrices de confusión de los tres modelos.

### 4.4 Prueba con Mensajes Nuevos

Para validar la aplicabilidad práctica, se probaron los modelos con 5 mensajes nuevos que no forman parte del dataset:

| # | Mensaje | Naive Bayes | SVM | Esperado |
|---|---|---|---|---|
| 1 | "Congratulations! You've won a FREE prize worth $1000. Click here to claim now!" | SPAM ✅ | SPAM ✅ | SPAM |
| 2 | "Hey, are we still meeting for lunch tomorrow?" | HAM ✅ | HAM ✅ | HAM |
| 3 | "URGENT: Your account will be suspended unless you verify your details immediately" | SPAM ✅ | HAM ⚠️ | SPAM |
| 4 | "Thanks for the notes from class. See you on Monday!" | HAM ✅ | HAM ✅ | HAM |
| 5 | "URGENT HOLAA NECESITAS DINERO YO PUEDO AYUDARTE" | SPAM ✅ | HAM ⚠️ | SPAM |

**Análisis:**
- El mensaje 3 es un intento de **phishing**, un tipo de spam poco representado en el dataset (que contiene sobre todo promociones y concursos). SVM no lo detecta; Naive Bayes sí, apoyándose en la palabra "urgent".
- El mensaje 5 está en **español**, idioma ausente del dataset: casi todas sus palabras son desconocidas para el modelo, y la clasificación depende únicamente de "urgent". Que Naive Bayes acierte es en buena parte casualidad, no evidencia de que funcione en español.

**Conclusión de pruebas:**
Los modelos funcionan correctamente con mensajes similares a los del dataset, pero son frágiles ante tipos de spam o idiomas que no vieron durante el entrenamiento.

### 4.5 Verificación de Hipótesis

**Hipótesis inicial:** "Se espera que SVM sea el mejor algoritmo para clasificación de spam"

**Resultado:** ❌ **NO CONFIRMADA** (SVM empata con Naive Bayes, pero no lo supera)

**Evidencia:**
1. En el conjunto de prueba, Naive Bayes obtiene un F1 de spam ligeramente mayor (0.9086 vs 0.9032)
2. En validación cruzada, la diferencia entre ambos (0.1 puntos de accuracy) es menor que la variación entre particiones
3. Ambos detectan exactamente la misma cantidad de spam (154 de 175)
4. SVM sí es superior cuando se evalúa con los duplicados incluidos, lo que explica la conclusión de la versión anterior del proyecto (ver 4.6)

### 4.6 Comparación con la Versión Anterior del Proyecto

Una primera versión del proyecto reportaba resultados distintos. Las diferencias se deben a tres problemas de metodología que se corrigieron:

| Modelo | Versión anterior | Corregida, con duplicados | **Corregida, sin duplicados** |
|---|---|---|---|
| SVM | 98.39% | 98.26% | **97.83%** |
| Naive Bayes | 96.71% | 98.14% | **97.96%** |
| kNN | 91.27% | 97.66% | **96.97%** |

**Problemas corregidos:**
1. **Fuga de información (*data leakage*)**: TF-IDF se ajustaba con todo el dataset, incluido el conjunto de prueba. Ahora se ajusta solo con el entrenamiento mediante `Pipeline`.
2. **Mensajes duplicados**: 501 mensajes repetidos podían quedar en entrenamiento y prueba al mismo tiempo, inflando los resultados (en especial los de SVM).
3. **División no estratificada**: la proporción de spam podía variar entre entrenamiento y prueba.

**Mejoras de configuración:**
- kNN con distancia coseno y voto ponderado (de 91.27% a 96.97%)
- Naive Bayes con `alpha=0.1` en lugar de 1.0 (suavizado excesivo para un vocabulario grande)
- TF-IDF con bigramas, TF sublineal y `min_df=2`

---

## 5. Discusión y Análisis

### 5.1 Interpretación de Resultados

#### ¿Por qué Naive Bayes y SVM empatan?

**Razones técnicas:**
1. **El problema es casi linealmente separable**: en el espacio TF-IDF, el spam se distingue por un conjunto reducido de términos muy característicos ("free", "txt", "claim", "prize", números cortos de servicio). Tanto SVM (una frontera lineal) como Naive Bayes (que también define una frontera lineal en el espacio logarítmico) separan bien estas clases.
2. **Naive Bayes funciona bien con textos cortos**: la asunción de independencia entre palabras se viola menos en mensajes de ~9 palabras que en documentos largos.
3. **El suavizado importa**: con `alpha=1.0` (valor por defecto), Naive Bayes asigna demasiada probabilidad a palabras no vistas y pierde recall de spam; con `alpha=0.1` alcanza a SVM.

**Razones empíricas:**
- Ambos modelos dejan pasar 21 mensajes de spam, y 15 de ellos son los mismos. Esto sugiere que la mayoría de los errores corresponden a mensajes intrínsecamente difíciles más que a debilidades de un algoritmo en particular.

#### ¿Por qué kNN queda en tercer lugar?

**Limitaciones identificadas:**
1. **Maldición de la dimensionalidad**: en un espacio de miles de dimensiones, la mayoría de los mensajes están "lejos" de todos los demás, y los vecinos más cercanos son menos informativos
2. **Desequilibrio de clases**: con una proporción de ~7.7:1 ham:spam, los vecinos tienden a ser ham, lo que reduce el recall de spam
3. **Costo computacional**: requiere comparar cada mensaje nuevo con todos los de entrenamiento

**El papel de la métrica de distancia:**
La mejora de 91.27% a 96.97% al cambiar de distancia euclidiana a coseno muestra que el bajo rendimiento anterior de kNN se debía principalmente a una métrica inadecuada para texto, no al algoritmo en sí.

#### Naive Bayes como opción práctica

**Ventajas prácticas:**
- **Rendimiento**: empata con SVM en este problema
- **Velocidad**: entrenamiento y predicción en milisegundos
- **Interpretabilidad**: las probabilidades por palabra permiten explicar por qué un mensaje se clasificó como spam

**Casos de uso ideales:**
- Sistemas con restricciones de recursos
- Aplicaciones en tiempo real
- Cuando se necesita explicar las decisiones

### 5.2 Análisis de Errores

#### Tipos de errores comunes:

**Falsos Negativos (Spam clasificado como Ham):**
- Mensajes de spam sofisticados que imitan lenguaje natural
- Tipos de spam poco representados en el dataset (por ejemplo, phishing)
- Técnicas de ofuscación (espacios extra, caracteres especiales, "fr33")
- Mensajes en otros idiomas

**Falsos Positivos (Ham clasificado como Spam):**
- Mensajes legítimos con palabras típicas de spam ("free", "urgent", "call")
- Promociones comerciales legítimas
- Alertas de seguridad genuinas

### 5.3 Limitaciones del Estudio

1. **Dataset específico**: Resultados en SMS pueden no generalizarse a email
2. **Idioma único**: Solo mensajes en inglés
3. **Temporalidad**: Patrones de spam evolucionan con el tiempo
4. **Hiperparámetros ajustados manualmente**: no se realizó una búsqueda sistemática (Grid Search)
5. **Conjunto de prueba pequeño para spam**: 175 mensajes, por lo que una diferencia de 1–2 aciertos cambia el F1 en ~0.5 puntos

### 5.4 Desequilibrio de Clases

**Impacto observado:**
- Ratio ~7.7:1 (ham:spam) tras eliminar duplicados
- Los modelos tienden a favorecer la clase mayoritaria
- El recall de spam (77–88%) es consistentemente menor que el recall de ham (99%)

**Medidas aplicadas:**
- División estratificada y validación cruzada estratificada
- Selección del mejor modelo por F1 de spam en lugar de accuracy

**Estrategias no implementadas:**
- Ajuste de pesos de clase (`class_weight="balanced"`)
- SMOTE (Synthetic Minority Over-sampling)
- Under-sampling de clase mayoritaria

---

## 6. Conclusiones

### 6.1 Hallazgos Principales

1. **Naive Bayes y SVM empatan como mejores algoritmos** (~98% de accuracy, F1 de spam ≈ 0.90). La hipótesis de que SVM sería superior no se confirma.

2. **Naive Bayes ofrece el mejor balance rendimiento-simplicidad**, siendo ideal para aplicaciones con restricciones de recursos o en tiempo real.

3. **kNN es viable con la métrica de distancia correcta**: con distancia coseno alcanza ~97% de accuracy, aunque sigue detectando menos spam que los otros dos modelos.

4. **La metodología de evaluación importa tanto como el algoritmo**: la fuga de información y los mensajes duplicados cambiaban el orden de los modelos y la conclusión del estudio.

5. **El preprocesamiento de texto es crucial**: la limpieza, el stemming y la configuración de TF-IDF son fundamentales para el éxito del sistema.

6. **Trade-off precision-recall**: Diferentes aplicaciones requerirán diferentes balances según la tolerancia a falsos positivos vs falsos negativos.

### 6.2 Contribuciones del Proyecto

**Académicas:**
- Comparación empírica de algoritmos clásicos con validación cruzada
- Demostración del efecto de la fuga de información y los duplicados en la evaluación
- Documentación detallada de metodología reproducible

**Prácticas:**
- Sistema funcional de detección de spam, con modelo guardable y reutilizable (`joblib`)
- Código reutilizable para futuros proyectos
- Guía de implementación para problemas similares

### 6.3 Aplicaciones Potenciales

1. **Filtrado de email**: Integración en clientes de correo
2. **Moderación de comentarios**: Redes sociales y foros
3. **Detección de phishing**: Sistemas de ciberseguridad
4. **Clasificación de mensajes**: Aplicaciones de mensajería
5. **Educación**: Herramienta didáctica para aprendizaje automático

---

## 7. Trabajo Futuro

### 7.1 Mejoras Inmediatas

1. **Optimización de hiperparámetros**
   - `GridSearchCV` para encontrar el k óptimo en kNN
   - Ajuste del parámetro C en SVM y de alpha en Naive Bayes
   - Comparar distintos rangos de n-gramas

2. **Balanceo de clases**
   - `class_weight="balanced"` en SVM para aumentar el recall de spam
   - Implementar SMOTE
   - Probar under-sampling

3. **Ingeniería de características**
   - N-gramas de caracteres (`analyzer="char_wb"`) para capturar ofuscaciones como "fr33" o "w1n"
   - Conservar números y símbolos de moneda (teléfonos, precios), que son buenos indicadores de spam
   - Características de longitud de mensaje y de patrones de puntuación

### 7.2 Extensiones Avanzadas

1. **Modelos ensemble**
   - Voting classifier combinando los tres modelos
   - Random Forest
   - Gradient Boosting

2. **Deep Learning**
   - Redes neuronales recurrentes (LSTM)
   - Transformers (BERT)
   - Word embeddings (Word2Vec, GloVe)

3. **Análisis multilingüe**
   - Entrenar modelos para español
   - Sistema multi-idioma

4. **Detección en tiempo real**
   - `HashingVectorizer` + `SGDClassifier` para entrenamiento incremental por lotes con memoria constante
   - Pipeline de procesamiento streaming

5. **Explicabilidad**
   - LIME/SHAP para interpretar predicciones
   - Visualización de palabras más influyentes

### 7.3 Validación Adicional

1. **Pruebas con otros datasets** (Enron Email, SpamAssassin, Spambase)
2. **Evaluación temporal**: entrenar con datos antiguos, probar con recientes
3. **Pruebas estadísticas** (por ejemplo, test de McNemar) para determinar si la diferencia entre modelos es significativa
4. **A/B testing** en ambiente de producción

---

## 8. Referencias Bibliográficas

### Datasets
- Almeida, T.A., Hidalgo, J.M.G., Yamakami, A. (2011). *SMS Spam Collection Dataset*. UCI Machine Learning Repository.

### Algoritmos
- Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine learning, 20(3), 273-297.
- McCallum, A., & Nigam, K. (1998). *A comparison of event models for naive bayes text classification*. AAAI-98 workshop on learning for text categorization.
- Cover, T., & Hart, P. (1967). *Nearest neighbor pattern classification*. IEEE transactions on information theory, 13(1), 21-27.

### Técnicas de Procesamiento
- Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information processing & management, 24(5), 513-523.
- Porter, M.F. (1980). *An algorithm for suffix stripping*. Program, 14(3), 130-137.

### Librerías Utilizadas
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural language processing with Python*. O'Reilly Media, Inc.

---

## 9. Apéndices

### Apéndice A: Especificaciones Técnicas

**Entorno de desarrollo:**
- Lenguaje: Python 3.10
- IDE: Visual Studio Code con extensión Jupyter
- Sistema Operativo: Windows
- Entorno virtual: `.venv` (dependencias en `requirements.txt`)

**Librerías principales:**
```
pandas==2.3.3
numpy==2.2.6
scikit-learn==1.7.2
nltk==3.10.4
joblib==1.6.0
matplotlib==3.10.9
seaborn==0.13.2
```

**Ejecución:**
```
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe clasificador_spam.py --cv 5
```

Opciones del script: `--datos` (archivo CSV o `SMSSpamCollection`), `--test` (proporción de prueba), `--cv` (particiones de validación cruzada), `--con-duplicados` y `--guardar` (guarda el mejor modelo con `joblib`).

### Apéndice B: Configuración de Modelos

**TF-IDF (común a los tres modelos):**
```python
TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, dtype=np.float32)
```

**Naive Bayes:**
```python
MultinomialNB(alpha=0.1)
```

**kNN:**
```python
KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
```

**SVM:**
```python
LinearSVC(C=1.0, random_state=42)
```

### Apéndice C: Estadísticas del Vocabulario

- **Palabras únicas tras limpieza**: 7,103
- **Términos en el vocabulario TF-IDF** (unigramas + bigramas con `min_df=2`, conjunto de entrenamiento): 4,799
- **Promedio de palabras por mensaje (tras limpieza)**: 8.9
- **Palabras más frecuentes en spam** (número de mensajes): "call" (240), "£" (180), "free" (135), "txt" (118), "text" (100)
- **Palabras más frecuentes en ham**: "u" (632), "im" (396), "go" (373), "get" (328), "come" (256)

Nota: el símbolo "£" no se elimina en la limpieza porque `string.punctuation` solo contiene signos ASCII; resulta ser uno de los indicadores más fuertes de spam (premios y tarifas en libras).

### Apéndice D: Tiempo de Ejecución

**Entrenamiento (3,545 mensajes, incluye la vectorización TF-IDF):**
- Naive Bayes: ~52 ms
- kNN: ~51 ms
- SVM: ~56 ms

**Predicción (1,520 mensajes):**
- Naive Bayes: ~16 ms
- kNN: ~69 ms
- SVM: ~16 ms

**Preprocesamiento completo:**
- Limpieza de texto: ~160 ms (la versión anterior del documento reportaba ~950 ms; la mejora se debe a precompilar las expresiones regulares y guardar en caché el resultado del stemming)

---

## 10. Guía para el Video Explicativo

### Estructura Sugerida (10-15 minutos)

**Introducción (1-2 min)**
- Presentación del equipo
- Planteamiento del problema del spam
- Objetivos del proyecto

**Marco Teórico (3-4 min)**
- Explicación visual de cada algoritmo:
  - Naive Bayes: Diagrama de probabilidades
  - kNN: Visualización de vecinos cercanos
  - SVM: Ilustración del hiperplano

**Metodología (3-4 min)**
- Pipeline de procesamiento con diagramas
- Demostración de limpieza de texto (ejemplo en vivo)
- Explicación de TF-IDF con ejemplo numérico
- División train/test

**Resultados (2-3 min)**
- Gráficos de comparación de modelos
- Matrices de confusión visualizadas
- Demo en vivo: clasificar mensajes nuevos

**Conclusiones (1-2 min)**
- Hallazgos principales
- Aplicaciones prácticas
- Trabajo futuro

**Elementos visuales recomendados:**
- Diapositivas con gráficos del notebook
- Screen recording de ejecución del código
- Animaciones de conceptos (hiperplano SVM)
- Ejemplos interactivos de clasificación

---

## Información del Proyecto

**Curso:** Recuperación de la Información  
**Institución:** [Tu institución]  
**Semestre:** 8vo  
**Fecha:** Noviembre 2025  

**Integrantes del equipo:**
1. [Nombre completo]
2. [Nombre completo]
3. [Nombre completo]
4. [Nombre completo]

---

*Este documento presenta el desarrollo completo del proyecto de clasificación de spam mediante algoritmos de aprendizaje supervisado, mostrando que Naive Bayes y Support Vector Machines alcanzan un rendimiento equivalente cuando la evaluación se realiza sin fuga de información.*
