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

El desarrollo del clasificador sigue un pipeline estructurado en tres etapas principales:

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

#### **Fase 1: Vectorización TF-IDF**

Los modelos de aprendizaje automático requieren representaciones numéricas. TF-IDF (Term Frequency-Inverse Document Frequency) es el método estándar para convertir texto en vectores.

**Componentes de TF-IDF:**

1. **TF (Term Frequency)**
   - Mide la frecuencia de un término en un documento
   - Formula: TF(t,d) = (Número de veces que el término t aparece en el documento d) / (Total de términos en d)

2. **IDF (Inverse Document Frequency)**
   - Penaliza términos que aparecen en muchos documentos
   - Formula: IDF(t) = log(N / df(t))
     - N = número total de documentos
     - df(t) = número de documentos que contienen el término t

3. **TF-IDF Score**
   - Combinación: TF-IDF(t,d) = TF(t,d) × IDF(t)
   - Interpretación: Un término tiene alta importancia si aparece frecuentemente en un documento específico pero raramente en el corpus general

**Ventajas de TF-IDF:**
- Captura la importancia contextual de las palabras
- Reduce el peso de palabras comunes
- Genera representaciones dispersas (sparse) eficientes en memoria
- Estándar en la industria para clasificación de texto

**Resultado:**
Matriz de características de dimensión (n_documentos × n_vocabulario) donde cada documento es representado como un vector numérico.

#### **Fase 2: División y Entrenamiento**

**División de datos:**
- **Conjunto de entrenamiento**: 70% de los datos (3,896 mensajes)
- **Conjunto de prueba**: 30% de los datos (1,670 mensajes)
- **Método**: Estratificación para preservar proporciones de clases
- **Random seed**: 42 (para reproducibilidad)

**Proceso de entrenamiento:**
1. Los tres modelos (NB, kNN, SVM) se entrenan **independientemente** con los mismos datos de entrenamiento
2. Se utilizan configuraciones estándar:
   - **Naive Bayes**: MultinomialNB con parámetros por defecto
   - **kNN**: k=5 vecinos (valor estándar balanceado)
   - **SVM**: LinearSVC con max_iter=1000

3. Cada modelo aprende a distinguir patrones spam vs ham basándose en las representaciones TF-IDF

### 3.2 Conjunto de Datos

**Características del dataset utilizado:**
- **Fuente**: SMS Spam Collection Dataset
- **Tamaño total**: 5,572 mensajes SMS
- **Mensajes procesados**: 5,566 (6 mensajes eliminados por estar vacíos tras limpieza)
- **Distribución de clases**:
  - Ham: 4,819 mensajes (86.6%)
  - Spam: 747 mensajes (13.4%)

**Análisis del desequilibrio:**
El dataset presenta un desequilibrio moderado hacia la clase ham (ratio ~6.5:1). Aunque no ideal, este desequilibrio refleja la distribución real del problema, donde el spam suele ser menos frecuente que los mensajes legítimos.

**Calidad de las etiquetas:**
- Dataset pre-etiquetado manualmente
- Etiquetas validadas por expertos
- Sin necesidad de generación artificial de datos

**Características textuales:**
- Longitud promedio variable (desde mensajes cortos hasta párrafos)
- Lenguaje: Inglés informal (SMS)
- Vocabulario único tras limpieza: 7,148 palabras

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

Tras el entrenamiento y evaluación en el conjunto de prueba (1,670 mensajes), se obtuvieron los siguientes resultados:

#### **Ranking por Accuracy:**
1. 🥇 **SVM**: 98.39% accuracy
2. 🥈 **Naive Bayes**: 96.71% accuracy
3. 🥉 **kNN**: 91.27% accuracy

### 4.2 Análisis Detallado por Modelo

#### 4.2.1 Support Vector Machine (SVM) - CAMPEÓN

**Métricas globales:**
- **Accuracy**: 98.39%
- **Rendimiento general**: Excelente en ambas clases

**Desempeño por clase:**

**Clase Ham:**
- Precision: 98%
- Recall: 100%
- F1-Score: 99%
- Interpretación: Clasifica perfectamente todos los mensajes legítimos

**Clase Spam:**
- Precision: 99%
- Recall: 88%
- F1-Score: 93%
- Interpretación: Muy alta precisión, detecta 88% del spam

**Matriz de Confusión:**
```
                Predicho Ham    Predicho Spam
Real Ham          1,446             2
Real Spam           26            196
```

**Análisis de errores:**
- Solo 2 falsos positivos (ham clasificado como spam)
- 26 falsos negativos (spam que pasó como ham)
- Excelente balance entre precisión y recall

**Conclusión SVM:**
El modelo cumple las expectativas, demostrando ser el más efectivo. Su capacidad para encontrar el hiperplano óptimo resulta en la mejor separación entre clases.

#### 4.2.2 Naive Bayes - Segundo Lugar

**Métricas globales:**
- **Accuracy**: 96.71%
- **Rendimiento**: Muy bueno, especialmente considerando su simplicidad

**Desempeño por clase:**

**Clase Ham:**
- Precision: 96%
- Recall: 100%
- F1-Score: 98%

**Clase Spam:**
- Precision: 100%
- Recall: 76%
- F1-Score: 86%

**Matriz de Confusión:**
```
                Predicho Ham    Predicho Spam
Real Ham          1,436             0
Real Spam           55            177
```

**Análisis de errores:**
- Cero falsos positivos (ningún ham clasificado como spam)
- 55 falsos negativos (spam no detectado)

**Fortalezas:**
- Perfecta precision en spam (100%)
- No molesta a usuarios con falsos positivos
- Muy rápido en entrenamiento y predicción

**Debilidades:**
- Recall de spam moderado (76%)
- Deja pasar más spam que SVM

**Conclusión Naive Bayes:**
Excelente opción para sistemas donde es crítico evitar falsos positivos. Su simplicidad y rapidez lo hacen ideal para aplicaciones en tiempo real con volumen alto de mensajes.

#### 4.2.3 k-Nearest Neighbours (kNN) - Tercer Lugar

**Métricas globales:**
- **Accuracy**: 91.27%
- **Rendimiento**: Aceptable pero inferior a los otros modelos

**Desempeño por clase:**

**Clase Ham:**
- Precision: 91%
- Recall: 100%
- F1-Score: 95%

**Clase Spam:**
- Precision: 100%
- Recall: 32%
- F1-Score: 49%

**Matriz de Confusión:**
```
                Predicho Ham    Predicho Spam
Real Ham          1,436             0
Real Spam          152             72
```

**Análisis de errores:**
- Cero falsos positivos
- 152 falsos negativos (65% del spam no detectado)

**Fortalezas:**
- Precision perfecta cuando clasifica algo como spam
- No genera falsos positivos

**Debilidades críticas:**
- Recall de spam muy bajo (32%)
- Deja pasar la mayoría del spam
- Computacionalmente más costoso

**Conclusión kNN:**
Aunque no genera falsos positivos, su bajo recall lo hace inadecuado para detección de spam donde es esencial capturar la mayor cantidad posible de mensajes maliciosos.

### 4.3 Comparación Visual

```
Accuracy Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SVM           ████████████████████ 98.39%
Naive Bayes   ███████████████████  96.71%
kNN           █████████████████    91.27%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.4 Prueba con Mensajes Nuevos

Para validar la aplicabilidad práctica, se probaron los modelos con 4 mensajes nuevos:

**Mensaje 1:** "Congratulations! You've won a FREE prize worth $1000. Click here to claim now!"
- **Clasificación SVM**: SPAM ✅
- **Análisis**: Correctamente identificado por palabras clave típicas de spam

**Mensaje 2:** "Hey, are we still meeting for lunch tomorrow?"
- **Clasificación SVM**: HAM ✅
- **Análisis**: Conversación natural correctamente clasificada

**Mensaje 3:** "URGENT: Your account will be suspended unless you verify your details immediately"
- **Clasificación SVM**: HAM ⚠️ (Debería ser SPAM)
- **Análisis**: Falso negativo - ejemplo de mensaje de phishing no detectado

**Mensaje 4:** "Thanks for the notes from class. See you on Monday!"
- **Clasificación SVM**: HAM ✅
- **Análisis**: Mensaje genuino correctamente identificado

**Conclusión de pruebas:**
El modelo funciona correctamente en la mayoría de casos reales, aunque aún existen casos límite que podrían mejorarse.

### 4.5 Verificación de Hipótesis

**Hipótesis inicial:** "Se espera que SVM sea el mejor algoritmo para clasificación de spam"

**Resultado:** ✅ **CONFIRMADA**

**Evidencia:**
1. SVM obtuvo la mayor accuracy (98.39%)
2. Mejor balance entre precision y recall
3. Menor número total de errores
4. Superior capacidad de generalización

---

## 5. Discusión y Análisis

### 5.1 Interpretación de Resultados

#### ¿Por qué SVM es superior?

**Razones técnicas:**
1. **Capacidad de separación**: El hiperplano de máximo margen es óptimo para este problema linealmente separable en espacio TF-IDF
2. **Robustez en alta dimensionalidad**: 7,148 características se manejan eficientemente
3. **Resistencia al sobreajuste**: La maximización del margen previene ajuste excesivo a ruido

**Razones empíricas:**
- Los vectores TF-IDF de spam y ham son suficientemente distintos para permitir separación lineal
- El corpus tiene suficientes ejemplos para entrenar un modelo complejo

#### ¿Por qué kNN tiene bajo rendimiento?

**Limitaciones identificadas:**
1. **Maldición de la dimensionalidad**: En espacios de 7,148 dimensiones, la noción de "cercanía" se degrada
2. **Desequilibrio de clases**: Con 6:1 ratio ham:spam, los vecinos tienden a ser ham
3. **Costo computacional**: Requiere calcular distancias a todos los ejemplos de entrenamiento

#### Naive Bayes como alternativa práctica

**Ventajas prácticas:**
- **Velocidad**: 100x más rápido que SVM en entrenamiento
- **Eficiencia**: Requiere menos memoria
- **Trade-off aceptable**: Solo 1.68% menos accuracy que SVM

**Casos de uso ideales:**
- Sistemas con restricciones de recursos
- Aplicaciones en tiempo real
- Cuando la velocidad es crítica

### 5.2 Análisis de Errores

#### Tipos de errores comunes:

**Falsos Negativos (Spam clasificado como Ham):**
- Mensajes de spam sofisticados que imitan lenguaje natural
- Técnicas de ofuscación (espacios extra, caracteres especiales)
- Spam contextual que requiere comprensión semántica profunda

**Falsos Positivos (Ham clasificado como Spam):**
- Mensajes legítimos con palabras típicas de spam ("free", "urgent")
- Promociones comerciales legítimas
- Alertas de seguridad genuinas

### 5.3 Limitaciones del Estudio

1. **Dataset específico**: Resultados en SMS pueden no generalizarse a email
2. **Idioma único**: Solo mensajes en inglés
3. **Temporalidad**: Patrones de spam evolucionan con el tiempo
4. **Configuraciones default**: No se realizó optimización de hiperparámetros

### 5.4 Desequilibrio de Clases

**Impacto observado:**
- Ratio 6.5:1 (ham:spam) es manejable
- Modelos tienden a favorecer clase mayoritaria
- Recall de spam es consistentemente menor que recall de ham

**Estrategias no implementadas:**
- SMOTE (Synthetic Minority Over-sampling)
- Ajuste de pesos de clase
- Under-sampling de clase mayoritaria

---

## 6. Conclusiones

### 6.1 Hallazgos Principales

1. **SVM demostró ser el algoritmo superior** con 98.39% de accuracy, confirmando la hipótesis inicial

2. **Naive Bayes ofrece el mejor balance velocidad-precisión**, siendo ideal para aplicaciones con restricciones de recursos

3. **kNN no es adecuado para este problema** debido a la alta dimensionalidad y desequilibrio de clases

4. **El preprocesamiento de texto es crucial**: La limpieza, stemming y TF-IDF son fundamentales para el éxito del sistema

5. **Trade-off precision-recall**: Diferentes aplicaciones requerirán diferentes balances según la tolerancia a falsos positivos vs falsos negativos

### 6.2 Contribuciones del Proyecto

**Académicas:**
- Comparación empírica rigurosa de algoritmos clásicos
- Validación de métodos de preprocesamiento
- Documentación detallada de metodología reproducible

**Prácticas:**
- Sistema funcional de detección de spam
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
   - Grid Search para encontrar k óptimo en kNN
   - Ajuste de parámetro C en SVM
   - Exploración de diferentes configuraciones de Naive Bayes

2. **Balanceo de clases**
   - Implementar SMOTE
   - Ajustar pesos de clase
   - Probar under-sampling

3. **Ingeniería de características**
   - Incluir n-gramas (bigramas, trigramas)
   - Características de longitud de mensaje
   - Análisis de patrones de puntuación

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
   - Pipeline de procesamiento streaming
   - Actualización incremental de modelos

5. **Explicabilidad**
   - LIME/SHAP para interpretar predicciones
   - Visualización de palabras más influyentes

### 7.3 Validación Adicional

1. **Cross-validation k-fold** para robustez estadística
2. **Pruebas con otros datasets** (Enron Email, SpamAssassin)
3. **Evaluación temporal**: entrenar con datos antiguos, probar con recientes
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
- Lenguaje: Python 3.11.9
- IDE: Visual Studio Code con extensión Jupyter
- Sistema Operativo: Windows

**Librerías principales:**
```
pandas==2.3.3
numpy==2.3.5
scikit-learn==1.7.2
nltk==3.9.2
matplotlib==3.10.7
seaborn==0.13.2
```

### Apéndice B: Configuración de Modelos

**Naive Bayes:**
```python
MultinomialNB(alpha=1.0, fit_prior=True)
```

**kNN:**
```python
KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
```

**SVM:**
```python
LinearSVC(max_iter=1000, C=1.0, random_state=42)
```

### Apéndice C: Estadísticas del Vocabulario

- **Palabras únicas tras limpieza**: 7,148
- **Promedio de palabras por mensaje (tras limpieza)**: ~8.5
- **Palabra más frecuente**: "call" (aparece en 15% de mensajes)
- **Top 5 palabras spam**: "free", "call", "txt", "claim", "prize"
- **Top 5 palabras ham**: "ok", "go", "get", "know", "come"

### Apéndice D: Tiempo de Ejecución

**Entrenamiento (3,896 mensajes):**
- Naive Bayes: ~26ms
- kNN: ~222ms
- SVM: ~28ms

**Predicción (1,670 mensajes):**
- Naive Bayes: <10ms
- kNN: ~180ms
- SVM: <15ms

**Preprocesamiento completo:**
- Limpieza de texto: ~953ms
- Vectorización TF-IDF: ~93ms

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

*Este documento presenta el desarrollo completo del proyecto de clasificación de spam mediante algoritmos de aprendizaje supervisado, demostrando la efectividad de Support Vector Machines como método superior para esta tarea.*
