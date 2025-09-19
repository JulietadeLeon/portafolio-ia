# Apuntes

## ¿Qué es Machine Learning?

![.](assets/image.png)


## 3 definiciones de Machine Learning

- Es una rama de la inteligencia artificial que permite a los sistemas aprender de los datos sin ser programados explícitamente. (Resumen Google)
- El machine learning es una rama de la inteligencia artificial (IA) centrada en entrenar a computadoras y máquinas para imitar el modo en que aprenden los humanos, realizar tareas de forma autónoma y mejorar su rendimiento y precisión a través de la experiencia y la exposición a más datos. (IBM)
- La capacidad de aprendizaje y de predicción de las máquinas se ha incrementado a lo largo del tiempo. Esto se observa tanto en los asistentes virtuales, que cada vez son más eficientes al responder y ejecutar tareas gracias a la mejora de los grandes modelos de lenguaje (LLM), como en plataformas de 'streaming' o comercio electrónico, que utilizan algoritmos para personalizar contenidos. El 'machine learning', especializado en el reconocimiento de patrones, es un campo en auge con décadas de historia. (BBVA)

## Diferencias con inteligencia artificial

- Inteligencia Artificial (IA): Es el concepto más amplio. Engloba todas las técnicas para que una máquina “piense” o actúe de manera inteligente, incluyendo reglas programadas a mano, lógica, búsqueda, y también machine learning.
- Machine Learning (ML): Es un subconjunto de la IA que se basa específicamente en que la máquina aprenda de datos y mejore con la experiencia, en lugar de seguir reglas fijas preprogramadas.

En resumen: Toda ML es IA, pero no toda IA es ML.

## Diferencias con análisis estadístico

En común:

- Ambos usan datos para encontrar patrones, relaciones y tendencias.
- Requieren técnicas matemáticas y conocimiento en probabilidad.
- Buscan apoyar la toma de decisiones basadas en evidencia.

Diferencias:

- Análisis estadístico: Se enfoca en describir, resumir e inferir conclusiones sobre un conjunto de datos, generalmente con hipótesis predefinidas y modelos interpretables. La prioridad es entender el fenómeno.
- Machine Learning: Se centra más en construir modelos predictivos o de clasificación que funcionen bien con datos nuevos, a veces sin necesidad de hipótesis previas, y prioriza la precisión de la predicción sobre la interpretabilidad.

En pocas palabras: el análisis estadístico explica y valida hipótesis; el machine learning predice y se adapta automáticamente a nuevos datos.

## Diferencias con Data Mining

En común:

- Tanto Machine Learning como Data Mining buscan descubrir patrones y conocimiento útil en grandes volúmenes de datos.
- Usan técnicas estadísticas, matemáticas y computacionales.
- Se aplican en ámbitos como marketing, finanzas, salud, seguridad, etc.

Diferencias:

- Data Mining: Es el proceso general de explorar y extraer conocimiento a partir de grandes bases de datos. Incluye tareas como segmentación, detección de anomalías, asociación de reglas y, muchas veces, usa machine learning como herramienta. El foco está en descubrir información previamente desconocida.
- Machine Learning: Es una rama específica (y más técnica) que desarrolla modelos que aprenden de datos para realizar predicciones o clasificaciones. No se limita a exploración; busca generalizar a nuevos datos.

## Herramientas o librerías de IA

- analizar y procesar datos tabulares
- entrenar modelos de machine learning
- procesar y entender texto
- entender texto o hacer chatbots

PyTorch

Torch es una biblioteca de aprendizaje automático de código abierto conocida por su gráfico computacional dinámico y favorecida por los investigadores. El marco es excelente para la creación de prototipos y la experimentación. Además, cuenta con el apoyo creciente de la comunidad, que ha creado herramientas como PyTorch. PyTorch se ha convertido rápidamente en uno de los frameworks más utilizados, útil en todo tipo de aplicaciones.

Keras

Keras es una API de redes neuronales de alto nivel de código abierto que se ejecuta sobre TensorFlow u otros marcos. Es fácil de usar y de aprender, lo que simplifica el proceso de trabajar con modelos de aprendizaje profundo. Además, es ideal para la creación rápida de prototipos. Sólo debes tener en cuenta que Keras puede carecer de algunas funciones avanzadas para tareas complejas.

TensorFlow

Desarrollado por Google, TensorFlow es una de las bibliotecas de IA de uso general más utilizadas. Ofrece un ecosistema flexible de herramientas, bibliotecas y recursos comunitarios para ayudar a los investigadores y desarrolladores a desarrollar e implementar una variedad de modelos de IA.

## Proceso CRISP-DM

- comprensión del negocio
- comprensión de datos
- preparación de datos
- modelado
- evaluación
- despliegue

[](https://lh7-rt.googleusercontent.com/docsz/AD_4nXdVQuMvGmZuhBAhoMgRjS40sD9GhilfvbMmbkgaykDxnbKuW1dvGUfHHXRC4dvqoLJUJtVTeFniwm3pit49yog_1_w0-Su6Wqjh5noNklP1DEQD8spOBtGhGKUKfLjyh4talsbdDA?key=FN5pMZSQ79RaqUo1B27ldQ)

De manera simple, Machine Learning consiste en enseñar a las computadoras a aprender patrones de datos con el fin de realizar predicciones.

Su funcionamiento es similar al del cerebro humano:

- Primero se observan muchos ejemplos.
- Luego se detectan patrones en esos datos.
- Finalmente, se hacen predicciones sobre casos nuevos.

## **Machine Learning**

El proceso de Machine Learning se basa en tres pasos principales:

1. Se alimentan datos al algoritmo.
2. El algoritmo encuentra patrones matemáticos en esos datos.
3. Con esos patrones, puede hacer predicciones sobre datos nuevos.

**Ejemplo:** después de analizar 1000 casas vendidas, el algoritmo es capaz de predecir el precio de una casa nueva.

![image.png](image%201.png)

## **EDA**

EDA significa Exploratory Data Analysis o en español Análisis Exploratorio de Datos.

Es una etapa fundamental dentro de un proyecto de análisis de datos o machine learning y consiste en:

- **Explorar los datos** para entender su estructura, calidad y características.
- **Detectar patrones, tendencias y relaciones** entre variables.
- **Identificar valores atípicos (outliers)**, datos faltantes o errores.
- **Usar estadísticas descriptivas** (medias, medianas, varianzas, distribuciones) y **visualizaciones** (gráficos, histogramas, diagramas de dispersión, etc.) para resumir la información.

En pocas palabras: un **EDA** es como “conocer tus datos” antes de aplicar modelos o algoritmos, asegurándote de entender qué información tienes y qué problemas puede haber.

```python
train.head(3)
```

- Muestra las **primeras filas** del dataset (por defecto 5).
- Útil para ver cómo están organizados los datos.

```python
train.info()
```

Resume la **estructura del dataset**: número de filas, columnas, tipos de datos y cuántos valores faltan en cada columna.

```python
train.describe(include='all').T
```

- Genera un **resumen estadístico** de todas las columnas (numéricas y categóricas).
- El “.T” lo transpone, dejando variables como filas para leerlo más fácil.

```python
train.isna().sum().sort_values(ascending=False)
```

- Cuenta la cantidad de **valores faltantes (NaN)** por columna.
- Luego los ordena de mayor a menor.

```python
train['Survived'].value_counts(normalize=True)
```

- Cuenta cuántos registros hay de cada categoría en la columna “survived”.
- Con “normalize=True” devuelve los **porcentajes** en lugar de los conteos.

## EDA Visual

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Supervivencia global por sexo
sns.countplot(data=train, x='Survived', hue='Sex', ax=axes[0,0])
axes[0,0].set_title('Supervivencia por sexo')

# Tasa de supervivencia por clase
sns.barplot(data=train, x='Pclass', y='Survived', estimator=np.mean, ax=axes[0,1])
axes[0,1].set_title('Tasa de supervivencia por clase')

# Distribución de edad por supervivencia
sns.histplot(data=train, x='Age', hue='Survived', kde=True, bins=30, ax=axes[1,0])
axes[1,0].set_title('Edad vs supervivencia')

# Correlaciones numéricas
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
sns.heatmap(train[numeric_cols].corr(), annot=True, cmap='Blues', ax=axes[1,1])
axes[1,1].set_title('Correlaciones')

plt.tight_layout()
plt.show()
```

### 📊 1. **Supervivencia por sexo** (`countplot`)

- Tipo: **Gráfico de barras**.
- Qué muestra: Cuántos hombres y cuántas mujeres sobrevivieron (Survived=1) y cuántos no (Survived=0).
- Interpretación: Sirve para comparar la **proporción de supervivencia entre sexos**. En el Titanic, se ve claramente que **sobrevivieron más mujeres que hombres**.

---

### 📊 2. **Tasa de supervivencia por clase** (`barplot`)

- Tipo: **Gráfico de barras con medias**.
- Qué muestra: El promedio de supervivencia (probabilidad de sobrevivir) según la clase del pasajero (`Pclass`).
- Interpretación: Generalmente, los pasajeros de **primera clase tuvieron más chances de sobrevivir** que los de segunda o tercera. Muestra cómo la **posición socioeconómica** influyó en la supervivencia.

---

### 📊 3. **Edad vs supervivencia** (`histplot`)

- Tipo: **Histograma con colores por supervivencia**.
- Qué muestra: La distribución de la edad de los pasajeros, separada en dos grupos: sobrevivientes (1) y no sobrevivientes (0).
- Interpretación: Se puede ver si **ciertas edades tenían más probabilidad de sobrevivir**. Por ejemplo, suelen notarse más **niños sobrevivientes** que adultos mayores, reflejando la política de "mujeres y niños primero".

---

### 📊 4. **Mapa de calor de correlaciones** (`heatmap`)

- Tipo: **Heatmap con anotaciones numéricas**.
- Qué muestra: Las correlaciones entre las variables numéricas (`Survived, Pclass, Age, SibSp, Parch, Fare`).
- Interpretación: Ayuda a detectar **relaciones lineales**.
    - Ejemplo: `Pclass` y `Fare` están negativamente correlacionados (a menor clase, mayor tarifa).
    - También se puede ver qué variables están más ligadas a `Survived`.

![image.png](image%202.png)

## Limpieza

## Feature Engineering

## Features finales

# Repaso TA4

## Diferencias claves

![image.png](image%203.png)

## Contaminacion de datos (Data Leakage)

Cuando el modelo "ve" información que NO debería tener durante el
entrenamiento
Como hacer trampa en un examen:
• Ver las respuestas antes del examen
• Solo estudiar el material permitido

Ejemplo: Data Leakage en Acción

**INCORRECTO:**

1. Preprocesar TODO el dataset
2. Dividir en train/test
3. Entrenar modelo

**CORRECTO**:

1. Dividir en train/test
2. Preprocesar SOLO con datos de entrenamiento
3. Aplicar preproces ado a test

**¿Por qué es tan Peligroso?**
• Optimismo artificial: Métricas infladas
• Información del futuro: El modelo "hace trampa"
• Falla en producción: Rendimiento real muy bajo
• Decisiones erróneas: Seleccionas modelo malo

**Solución: Pipelines**
Pipeline = Secuencia automática de pasos
• Previene data leakage automáticamente
• Aplica transformaciones en orden correcto
• Garantiza proceso robusto

## Validacion cruzada (Cross-Validation)

**Problema del Train/Test Split**
• Una sola división = Una sola "opinión"
• Resultados pueden variar por suerte
• ¿El modelo es bueno o tuvo suerte?
Necesitamos múltiples "opiniones”

**¿Qué es Cross-Validation?**
• Dividir datos en K partes (folds)
• Entrenar K veces, cada vez con diferente test
• Promedio de K resultados = estimación robusta
• Desviación estándar = estabilidad del modelo

![image.png](image%204.png)

## Comparación de Modelos

**¿Por qué Comparar Modelos?**

• Diferentes algoritmos para diferentes problemas
• No hay "modelo perfecto universal"
• Competencia revela el mejor para TUS datos
• Combinar rendimiento + estabilidad

**Candidatos Típicos**

• Logistic Regression: Simple, interpretable
• Ridge Classifier: Con regularización
• Random Forest: Ensemble, robusto
• SVM: Fronteras complejas

**Proceso de Comparación**

1. Crear pipelines para cada modelo
2. Evaluar con cross-validation
3. Comparar accuracy promedio
4. Analizar estabilidad (desviación)
5. Seleccionar ganador

**Métricas de Selección**

• Rendimiento: ¿Cuál es más preciso?
• Estabilidad: ¿Cuál es más consistente?
• Velocidad: ¿Cuál entrena/predice más rápido?
• Memoria: ¿Cuál usa menos recursos?
• Interpretabilidad: ¿Cuál es más explicable?

# Problemas de escala

![image.png](image%205.png)

# Normalización

Normalizar = poner todas las columnas en la misma escala para comparar
manzanas con manzanas.

- Hace que las variables hablen el mismo idioma.
- Ayuda a los algoritmos a aprender más parejo.
- No crea información nueva ni arregla sesgos.

# Min-Max vs StandardScaler

Dos formas rápidas de “ajustar el volumen”.
**Min-Max (0–1) → “aprieta y estira” para que el más chico sea 0 y el más grande 1.**

- Datos limpios y acotados.
- Con outliers se rompe (todo queda aplastado).

**StandardScaler (Z-score) → “centra en 0 y ajusta el tamaño”.**

- Mejor cuando hay valores raros.
- Suele ayudar a optimizadores basados en gradiente.

![image.png](image%206.png)

# Principal Component Analysis (PCA)

PCA es un resumen inteligente: te deja mirar muchos datos a la vez desde el mejor ángulo.

- Visualizar en 2D/3D datos que tienen decenas de columnas.
- Acelerar modelos (menos columnas = menos cuentas).
- Quitar ruido y duplicaciones (variables muy parecidas entre sí)

Imaginá una nube de puntos. Giramos el plano para mirar por la dirección donde la nube cambia más (PC1). La segunda mejor dirección, en 90°, es PC2.

![image.png](image%207.png)

**Cómo se calcula cada PC**

- Preparar los datos: escalá y centrá (media 0) cada columna usando solo TRAIN.
- Buscar la dirección con más variación: el algoritmo mira la nube y encuentra la flecha por donde los puntos están más esparcidos → esa es PC1.
- Forzar 90°: ahora busca otra flecha perpendicular (90°) que capture la mayor variación restante → PC2.
- Repite para PC3, PC4, … siempre en 90° con las anteriores (por eso las PCs son
independientes entre sí).

# **PCA: Varianza explicada**

“varianza explicada” = qué porcentaje del dibujo original conserva tu resumen.

¿Contra qué datos se calcula? Contra los datos usados para entrenar el PCA
(normalizados/centrados). No contra todo el dataset: primero fit en TRAIN, luego transform
en VALID/TEST (para evitar leakage).

**Cómo se mide (idea simple):**

- Cada PC trae un valor de varianza (cuánto “cuenta”).
- El ratio de cada PC = varianza de esa PC / varianza total.
- La acumulada te dice cuánta info llevás con las primeras k PCs.

¿Con cuántos componentes? Reglas prácticas

- Visualizar: 2–3 PCs (solo para mirar).
- Compactar/Acelerar: elegí k tal que la acumulada sea 90–95%

# Selección de atributos

PCA hace un resumen mezclando columnas; la selección de atributos elige las
mejores columnas y descarta el resto.

- Filtro (rápidos): quitá columnas casi constantes (variance threshold) y columnas duplicadas/muy correlacionadas.
- Wrappers (prueba y error): el modelo te dice qué conviene.
    - Forward: empezás con 0 y vas sumando columnas.
    - Backward: empezás con todas y vas sacando.
    - Usá una métrica para decidir
- Embebidos (lo decide el modelo):
    - L1/Lasso: deja varias en cero (se “apagan”).
    - Árboles/Random Forest: traen importancia de variables.
    - Permutation importance: medí cuánto empeora si desordenás una columna.

# Clustering =/= Clasificación

- Supervisado = examen con respuestas (entrenás con etiquetas).
- No supervisado = explorás y buscás grupos sin respuestas.
- En clustering no hay “verdad”: hay segmentos útiles.
- Métricas internas (Silhouette) ayudan, pero negocio manda.

# K-Means: espacio, posición y distancia

![image.png](image%208.png)

**¿Qué es el espacio?**

- Pensá que cada columna (ya escalada) es un eje.
- Si tenés 3 columnas (Edad, Ingresos, Tiempo), vivís en un espacio 3D. Con
20 columnas, es 20D (no lo vemos, pero existe matemáticamente).

**¿Qué significa la posición de un punto?**

- Es simplemente su vector de valores en esos ejes: x = [edad_est, ingresos_est, tiempo_est, …].
- Por eso escalar es clave: si un eje tiene números gigantes, domina la
posición y las distancias.

**¿Qué es el centroide?**

- Es el promedio componente a componente de los puntos del cluster:
    - μ = mean( x₁, x₂, …, x_n ).
- Intuición: el punto “típico” del grupo.

**¿Cómo mide la distancia?**

- K-Means clásico usa distancia euclídea (la de regla): “qué tan lejos” está el punto del centroide sumando cuadraditos en cada eje.
- El objetivo que minimiza es la suma de distancias al cuadrado (SSE).
Por eso el mejor centroide resulta ser el promedio (no la mediana)