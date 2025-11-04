---
title: "Lecturas"
date: 2025-01-01
---

# Aprendizajes al completar UT2: 

• Comprender la arquitectura de perceptrones multicapa y funciones de activación

• Desarrollar MLPs avanzados usando PyTorch Lightning para aplicaciones reales

• Aplicar técnicas de optimización (SGD, AdamW) y entender backpropagation

• Implementar técnicas de regularización y visualización con TensorBoard/Mlflow

• Experimentar con optimizadores avanzados y learning rate scheduling

# Lecturas

## **Lecturas minimas (Evaluacion el 16/09):**

### **Kaggle Intro to Deep Learning (Completo):**

- [A Single Neuron](https://www.kaggle.com/code/ryanholbrook/a-single-neuron)
- [Deep Neural Networks](https://www.kaggle.com/code/ryanholbrook/deep-neural-networks)
- [Stochastic Gradient Descent](https://www.kaggle.com/code/ryanholbrook/stochastic-gradient-descent)
- [Binary Classification](https://www.kaggle.com/code/ryanholbrook/binary-classification)
- [Dropout and Batch Normalization](https://www.kaggle.com/code/ryanholbrook/dropout-and-batch-normalization)
- [Overfitting and Underfitting](https://www.kaggle.com/code/ryanholbrook/overfitting-and-underfitting)

### **Google Deep Learning:**

- [Neural Networks Course](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook)

### **PyTorch Lightning:**

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Getting Started Guide](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

## **Lecturas totales:**

### **Kaggle Intro to Deep Learning (Completo):**

- [A Single Neuron](https://www.kaggle.com/code/ryanholbrook/a-single-neuron)
- [Deep Neural Networks](https://www.kaggle.com/code/ryanholbrook/deep-neural-networks)
- [Stochastic Gradient Descent](https://www.kaggle.com/code/ryanholbrook/stochastic-gradient-descent)
- [Binary Classification](https://www.kaggle.com/code/ryanholbrook/binary-classification)
- [Dropout and Batch Normalization](https://www.kaggle.com/code/ryanholbrook/dropout-and-batch-normalization)
- [Overfitting and Underfitting](https://www.kaggle.com/code/ryanholbrook/overfitting-and-underfitting)

### **Google Deep Learning:**

- [Neural Networks Course](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook)
- [Deep Learning Tuning Playbook](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook)

### **PyTorch Ecosystem:**

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Getting Started Guide](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)

## **Herramientas:**

### **Fundamentals:**

- NumPy Documentation: https://numpy.org/doc/stable/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/

### **Optimización:**

- PyTorch Optimizers: https://pytorch.org/docs/stable/optim.html
- Learning Rate Scheduling: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

### **Visualización:**

- TensorBoard Documentation: https://www.tensorflow.org/tensorboard
- Matplotlib Documentation: https://matplotlib.org/stable/contents.html

### **Data Handling:**

- PyTorch DataLoader: https://pytorch.org/docs/stable/data.html
- PyTorch Transforms: https://pytorch.org/vision/stable/transforms.html


# Resumen lecturas mínimas
# **Kaggle Intro to Deep Learning:**

## [A Single Neuron](https://www.kaggle.com/code/ryanholbrook/a-single-neuron)

## 📌 Resumen general

El notebook explica cómo funciona una neurona artificial:

- Recibe **entradas (features)**.
- Multiplica cada entrada por un **peso**.
- Suma todo y le agrega un **sesgo (bias)**.
- Pasa el resultado por una **función de activación** que introduce no linealidad.

Matemáticamente, la operación de una neurona se puede expresar como:

y=f(w1x1+w2x2+⋯+wnxn+b)y = f(w_1x_1 + w_2x_2 + \dots + w_nx_n + b)

y=f(w1x1+w2x2+⋯+wnxn+b)

donde:

- xix_ixi = característica de entrada,
- wiw_iwi = peso asociado,
- bbb = bias,
- fff = función de activación,
- yyy = salida de la neurona.

---

## 📖 Conceptos desarrollados en el notebook

### 1. **Features (Características de entrada)**

Son las variables de entrada que describen a un ejemplo de datos.

Ejemplo: para predecir el precio de una casa, podrían ser tamaño, número de habitaciones, ubicación.

En la neurona, cada feature se multiplica por un peso.

---

### 2. **Pesos (Weights)**

Son los parámetros que la red aprende durante el entrenamiento.

Indican cuánto contribuye cada feature en la predicción.

- Si un peso es grande y positivo → la feature aumenta el valor de salida.
- Si es negativo → lo disminuye.

---

### 3. **Sesgo (Bias)**

Es un valor adicional que desplaza la salida de la neurona.

Permite que la red se ajuste incluso cuando todas las features valen 0.

👉 Es como un intercepto en una regresión lineal.

---

### 4. **Función de activación**

Clave para introducir **no linealidad**.

Sin activación, la neurona sería solo una regresión lineal.

Algunas funciones comunes:

- **ReLU (Rectified Linear Unit):** devuelve 0 si la entrada es negativa, y la entrada misma si es positiva. Muy usada en deep learning.
- **Sigmoid:** convierte la salida en un valor entre 0 y 1. Útil para clasificación binaria.
- **tanh:** valores entre -1 y 1.

---

### 5. **Composición de neuronas → Red Neuronal**

El notebook señala que si apilamos muchas neuronas y varias capas, obtenemos una **red neuronal profunda**.

Cada capa transforma la representación de los datos.

---

### 6. **Analogía biológica**

Se menciona la inspiración en el cerebro humano:

- Las neuronas biológicas reciben señales a través de dendritas.
- Procesan esas señales y transmiten impulsos eléctricos a otras neuronas.
- En una red neuronal artificial, las conexiones se representan con pesos.

---

### 7. **Ejemplo práctico en Keras**

El notebook muestra cómo implementar una red neuronal de **una sola neurona** usando Keras:

```python
from tensorflow import keras
from tensorflow.keras import layers

# Modelo secuencial con una sola capa densa (una neurona)
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])

```

- `Dense(units=1)` → define una capa totalmente conectada con 1 neurona.
- `input_shape=[3]` → indica que la entrada tiene 3 features.

El modelo calcula:

output=f(w1x1+w2x2+w3x3+b)output = f(w_1x_1 + w_2x_2 + w_3x_3 + b)

output=f(w1x1+w2x2+w3x3+b)

---

### 8. **Entrenamiento de la neurona**

La neurona no "sabe" los pesos inicialmente, se entrenan con un **algoritmo de optimización** (descenso de gradiente).

- El modelo predice un valor.
- Se mide el error (función de pérdida).
- Se ajustan los pesos para minimizar ese error.

---

## 🚀 Conclusión

El notebook **“A Single Neuron”** es una introducción al bloque fundamental de cualquier red neuronal: la **neurona artificial**.

- Explica las piezas básicas: features, pesos, bias, activación.
- Da intuición matemática y biológica.
- Muestra un ejemplo práctico en Keras.
- Prepara el terreno para entender redes más profundas en el siguiente notebook.

## [Deep Neural Networks](https://www.kaggle.com/code/ryanholbrook/deep-neural-networks)

## 📌 Resumen general

En este notebook se pasa de la **neurona única** al concepto de **red neuronal profunda** (*deep neural network*).

La idea es:

- En lugar de tener una sola neurona que aprende una función lineal, usamos varias capas de neuronas.
- Esto permite que el modelo **aprenda representaciones más complejas y no lineales** de los datos.
- Introduce conceptos como **capas ocultas, funciones de activación, arquitectura de la red y Keras Sequential API**.

---

## 📖 Conceptos desarrollados

### 1. **Capa oculta (Hidden Layer)**

- Una capa que no es ni de entrada ni de salida.
- Toma los valores de la capa anterior, los transforma mediante pesos, bias y función de activación, y los pasa a la siguiente.
- Cada capa oculta aprende representaciones intermedias de los datos.

👉 Ejemplo:

- Entrada: tamaño de la casa, número de habitaciones.
- Capa oculta: puede aprender “nivel de lujo”.
- Otra capa: puede aprender “precio esperado”.

---

### 2. **Profundidad de la red (Deep)**

- Una red se considera “profunda” si tiene **más de una capa oculta**.
- Cada capa puede captar **patrones más abstractos**.
- Cuantas más capas → más capacidad de aprendizaje, pero también más riesgo de **overfitting** y mayor dificultad en entrenamiento.

---

### 3. **Funciones de activación**

Ya vistas en “Single Neuron”, pero aquí se destaca la importancia de usar funciones no lineales en capas ocultas:

- **ReLU (Rectified Linear Unit):** la más común en deep learning, permite que la red aprenda relaciones no lineales.
- Sin activaciones, incluso una red profunda sería equivalente a una regresión lineal.

---

### 4. **Arquitectura de la red**

- Especifica:
    - Número de capas ocultas.
    - Número de neuronas por capa.
    - Función de activación usada en cada capa.
- No hay una única regla fija → depende de la complejidad de los datos y del problema.

---

### 5. **Implementación en Keras**

El notebook enseña a construir una red con varias capas usando **`Sequential`**:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(units=512, activation="relu", input_shape=[784]),
    layers.Dense(units=256, activation="relu"),
    layers.Dense(units=10, activation="softmax")
])

```

- `Dense(units, activation)` → crea una capa densa con número de neuronas = `units`.
- `input_shape=[784]` → define la forma de la entrada (ejemplo clásico: imágenes MNIST de 28x28 = 784 pixeles).
- Primera capa oculta: 512 neuronas con ReLU.
- Segunda capa oculta: 256 neuronas con ReLU.
- Capa de salida: 10 neuronas con **softmax** (para clasificación multiclase).

---

### 6. **Softmax en la capa de salida**

- Convierte las salidas en probabilidades.
- La suma de todas las probabilidades es = 1.
- Se usa para problemas de clasificación multiclase.

---

### 7. **Expresividad de las redes profundas**

- Teóricamente, una red con una sola capa oculta y suficientes neuronas puede aproximar cualquier función (Teorema de aproximación universal).
- Pero en la práctica, **varias capas pequeñas suelen ser más eficientes** que una sola capa enorme.
- Las capas sucesivas **aprenden jerarquías de representación**:
    - Capas bajas: patrones simples (bordes en imágenes).
    - Capas medias: combinaciones de patrones (formas).
    - Capas altas: conceptos más abstractos (caras, números).

---

## 🚀 Conclusión

El notebook **“Deep Neural Networks”** introduce la idea de **profundidad**:

- Más capas y neuronas permiten que la red aprenda funciones más complejas.
- La clave está en elegir bien la arquitectura (número de capas, neuronas y activaciones).
- Muestra cómo implementar estas redes en **Keras** de forma simple con `Sequential` y `Dense`.

👉 Este paso conecta el modelo más simple (una sola neurona) con arquitecturas capaces de resolver problemas reales como **clasificación de imágenes o texto**.

## [Stochastic Gradient Descent](https://www.kaggle.com/code/ryanholbrook/stochastic-gradient-descent)

## 📌 Resumen general

Este notebook explica **cómo aprenden las redes neuronales**.

El foco está en el **algoritmo de optimización más usado en deep learning: Stochastic Gradient Descent (SGD)**.

- El objetivo es **ajustar los pesos y bias** de la red para minimizar el error.
- Se introduce la **función de pérdida (loss function)** como medida del error.
- Se explica cómo el **gradiente** indica la dirección de mayor descenso de la pérdida.
- Se diferencia entre **batch gradient descent, mini-batch y stochastic**.

---

## 📖 Conceptos desarrollados

### 1. **Función de pérdida (Loss Function)**

- Mide la diferencia entre la predicción del modelo y el valor real.
- El entrenamiento busca **minimizar la pérdida**.
- Ejemplos:
    - **MSE (Mean Squared Error):** para regresión.
    - **Cross-Entropy Loss:** para clasificación.

👉 Una buena elección de loss depende del tipo de problema.

---

### 2. **Descenso de gradiente (Gradient Descent)**

- Método matemático para encontrar el mínimo de una función.
- Idea: mover los parámetros www en la dirección contraria al gradiente de la función de pérdida.

wnuevo=wviejo−η⋅∇L(w)w_{nuevo} = w_{viejo} - \eta \cdot \nabla L(w)

wnuevo=wviejo−η⋅∇L(w)

donde:

- η\etaη = learning rate (tasa de aprendizaje),
- ∇L(w)\nabla L(w)∇L(w) = gradiente de la pérdida respecto a los pesos.

---

### 3. **Learning Rate (Tasa de aprendizaje)**

- Hiperparámetro clave.
- Define qué tan grandes son los pasos en cada actualización.
    - Muy alto → saltos grandes, el modelo no converge.
    - Muy bajo → avanza muy lento, puede atascarse en mínimos locales.

---

### 4. **Batch Gradient Descent**

- Calcula el gradiente usando **todos los datos del dataset** en cada paso.
- Ventaja: cálculo exacto del gradiente.
- Desventaja: muy lento y costoso con grandes datasets.

---

### 5. **Stochastic Gradient Descent (SGD)**

- Usa **un solo ejemplo** para calcular el gradiente en cada paso.
- Ventaja: rápido y más eficiente con datasets grandes.
- Desventaja: más “ruidoso”, la pérdida fluctúa en lugar de disminuir suavemente.
- Este ruido puede ser positivo → ayuda a escapar de mínimos locales.

---

### 6. **Mini-Batch Gradient Descent**

- Compromiso entre batch y SGD.
- Calcula el gradiente con un **subconjunto pequeño (mini-batch)** de ejemplos.
- Es el método más usado en práctica.

---

### 7. **Implementación en Keras**

El notebook muestra cómo definir el optimizador al compilar el modelo:

```python
from tensorflow import keras
from tensorflow.keras import layers

# Modelo simple
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])

# Compilar con SGD
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss="mean_squared_error"
)

```

- `optimizer=SGD` → usa descenso de gradiente estocástico.
- `learning_rate=0.01` → define el tamaño de los pasos.
- `loss="mse"` → mide el error entre predicciones y valores reales.

---

### 8. **Curvas de entrenamiento**

- El notebook suele mostrar cómo evoluciona la **pérdida (loss)** durante el entrenamiento.
- Con learning rate adecuado → la pérdida disminuye suavemente.
- Con learning rate inadecuado → puede oscilar demasiado o no mejorar.

---

## 🚀 Conclusión

El notebook **“Stochastic Gradient Descent”** enseña el mecanismo central del entrenamiento:

- Definir una función de pérdida.
- Usar gradientes para ajustar pesos y bias.
- Diferenciar entre **batch, stochastic y mini-batch gradient descent**.
- Entender el rol crítico del **learning rate**.

👉 Este paso conecta la teoría de las neuronas con el **proceso real de aprendizaje automático**.

## [Binary Classification](https://www.kaggle.com/code/ryanholbrook/binary-classification)

## 📌 Resumen general

Este notebook muestra cómo aplicar una red neuronal al problema de **clasificación binaria** (dos clases posibles, ej. “sí/no”, “spam/no spam”).

Los puntos clave son:

- Cómo se formula la salida para que represente una **probabilidad**.
- Qué **función de pérdida** se usa en clasificación binaria.
- Cómo interpretar la métrica **accuracy**.
- Cómo usar **sigmoid en la salida** y **binary cross-entropy** como pérdida.

---

## 📖 Conceptos desarrollados

### 1. **Clasificación binaria**

- Problema donde la etiqueta solo puede ser **0 o 1**.
- Ejemplos:
    - ¿Un email es spam?
    - ¿Una imagen es gato o no?
    - ¿Un cliente cancelará su suscripción?

---

### 2. **Unidad de salida con activación sigmoide**

- Se usa una **sola neurona de salida**.
- Su activación es **sigmoid** para producir valores en [0,1][0,1][0,1].
- Ese valor se interpreta como la **probabilidad de que la clase sea 1**.
- Decisión final: aplicar un **umbral**, típicamente 0.5.

---

### 3. **Función de pérdida: Binary Cross-Entropy (Log Loss)**

- Mide cuán buena es la predicción de probabilidades.
- Fórmula:

L(y,y^)=−[y⋅log⁡(y^)+(1−y)⋅log⁡(1−y^)]L(y, \hat{y}) = - \big[ y \cdot \log(\hat{y}) + (1-y) \cdot \log(1-\hat{y}) \big]

L(y,y^)=−[y⋅log(y^)+(1−y)⋅log(1−y^)]

donde:

- yyy = etiqueta real (0 o 1),
- y^\hat{y}y^ = probabilidad predicha.

👉 Penaliza mucho si el modelo asigna **baja probabilidad a la clase correcta**.

---

### 4. **Métrica: Accuracy**

- Accuracy = proporción de ejemplos bien clasificados.
- Se obtiene comparando la predicción umbralizada (ej. y^>0.5\hat{y} > 0.5y^>0.5) con la etiqueta real.
- Buena como métrica de desempeño, pero no sirve como función de pérdida porque no es diferenciable.

---

### 5. **Implementación en Keras**

Ejemplo típico en el notebook:

```python
from tensorflow import keras
from tensorflow.keras import layers

# Modelo simple para clasificación binaria
model = keras.Sequential([
    layers.Dense(units=16, activation="relu", input_shape=[num_features]),
    layers.Dense(units=1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

```

- Capa oculta: 16 neuronas con ReLU.
- Capa de salida: 1 neurona con **sigmoid**.
- Loss: **binary_crossentropy**.
- Métrica: **accuracy**.

---

### 6. **Interpretación de la probabilidad**

- La salida de sigmoid puede leerse como:
    - 0.85 → 85% de probabilidad de clase positiva.
    - 0.2 → 20% de probabilidad de clase positiva.
- Luego se aplica un umbral para convertirlo en una clase.

---

## 🚀 Conclusión

El notebook **“Binary Classification”** muestra la forma estándar de resolver un problema de 2 clases con deep learning:

- **Una sola salida con sigmoid.**
- **Pérdida = binary cross-entropy.**
- **Métrica = accuracy.**

👉 Prepara el camino para problemas de clasificación más complejos, como **multiclase con softmax**.

## [Dropout and Batch Normalization](https://www.kaggle.com/code/ryanholbrook/dropout-and-batch-normalization)

## 📌 Resumen general

Este notebook introduce **dos técnicas clave** para mejorar el entrenamiento de redes profundas:

1. **Dropout** → regularización que previene **overfitting** apagando neuronas al azar durante el entrenamiento.
2. **Batch Normalization (BatchNorm)** → normaliza activaciones dentro de cada capa para estabilizar y acelerar el entrenamiento.

Ambas técnicas ayudan a que la red **generalice mejor** y sea más **estable**.

---

## 📖 Conceptos desarrollados

### 1. **Overfitting**

- Problema cuando la red se ajusta demasiado al set de entrenamiento y pierde capacidad de generalizar.
- Dropout y BatchNorm son herramientas para combatirlo.

---

### 2. **Dropout**

- Técnica de regularización propuesta por Hinton (2014).
- Durante el entrenamiento, cada neurona tiene una **probabilidad ppp** de ser “apagada” (output = 0).
- Esto fuerza a la red a no depender de neuronas individuales y promueve **representaciones más robustas**.
- En inferencia (predicción real), no se apagan neuronas: en su lugar, los pesos se escalan para compensar.

👉 Ejemplo en Keras:

```python
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),   # 30% de las neuronas apagadas durante entrenamiento
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

```

---

### 3. **Batch Normalization**

- Problema: las activaciones en redes profundas pueden volverse inestables (cambian de escala y distribución entre capas).
- Solución: **normalizar** las activaciones dentro de cada mini-batch:

x^=x−μσ\hat{x} = \frac{x - \mu}{\sigma}

x^=σx−μ

donde μ\muμ y σ\sigmaσ son la media y desviación estándar del batch.

- Además, BatchNorm introduce parámetros aprendibles (γ,β\gamma, \betaγ,β) para re-escalar y desplazar después de la normalización.
- Beneficios:
    - Acelera el entrenamiento.
    - Permite usar learning rates más altos.
    - Tiene cierto efecto de regularización.

👉 Ejemplo en Keras:

```python
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(1, activation="sigmoid")
])

```

---

### 4. **Diferencias clave**

- **Dropout**: combate directamente el **overfitting** apagando neuronas al azar.
- **BatchNorm**: estabiliza y acelera el **entrenamiento** normalizando activaciones.
- Se pueden usar **juntas** en la misma red.

---

## 🚀 Conclusión

El notebook **“Dropout and Batch Normalization”** enseña dos técnicas muy usadas en práctica:

- **Dropout** → previene overfitting al introducir aleatoriedad.
- **BatchNorm** → estabiliza y acelera el entrenamiento al normalizar activaciones.

👉 Con estas herramientas, las redes profundas se vuelven más **robustas, rápidas y generalizables**.

## [Overfitting and Underfitting](https://www.kaggle.com/code/ryanholbrook/overfitting-and-underfitting)

## 📌 Resumen general

Este notebook trata uno de los problemas centrales en machine learning y deep learning:

- **Underfitting** → el modelo no aprende lo suficiente.
- **Overfitting** → el modelo aprende demasiado (memoriza el entrenamiento) y no generaliza bien.

Muestra cómo detectarlos y qué técnicas aplicar para mejorar el desempeño.

---

## 📖 Conceptos desarrollados

### 1. **Underfitting**

- El modelo es **demasiado simple** o no ha entrenado lo suficiente.
- No logra capturar los patrones de los datos.
- Síntomas:
    - Alto error en entrenamiento y validación.
    - Learning curves que no bajan.

👉 Causas comunes:

- Muy pocas capas/neuronas.
- Entrenamiento insuficiente (pocas épocas).
- Learning rate inadecuado.

---

### 2. **Overfitting**

- El modelo es **demasiado complejo** o entrenó demasiado tiempo.
- Aprende ruido o particularidades del set de entrenamiento.
- Síntomas:
    - Muy bajo error en entrenamiento.
    - Alto error en validación.
    - Divergencia entre las curvas de entrenamiento y validación.

👉 Causas comunes:

- Muchas capas/neuronas sin regularización.
- Dataset muy chico.
- Entrenamiento demasiado largo.

---

### 3. **Cómo diagnosticar: Learning Curves**

- Gráficos de **pérdida/accuracy en entrenamiento y validación**.
- Patrón típico:
    - Underfitting → ambas curvas altas (mal desempeño en todo).
    - Overfitting → entrenamiento muy bajo, validación alta (divergencia).
    - Buen ajuste → ambas curvas bajas y cercanas.

---

### 4. **Técnicas para combatir underfitting**

- Usar una **red más grande** (más capas o neuronas).
- Entrenar más tiempo (más épocas).
- Ajustar el **learning rate**.
- Revisar la arquitectura del modelo.

---

### 5. **Técnicas para combatir overfitting**

- Usar más **datos de entrenamiento** (data augmentation).
- Aplicar **regularización**:
    - Dropout.
    - L1/L2 penalties.
- Usar **Batch Normalization**.
- Parar antes de que empiece a memorizar (**early stopping**).

---

### 6. **Implementación en Keras**

Ejemplo clásico del notebook:

```python
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

```

- Dropout ayuda a controlar el overfitting.
- Early stopping se puede agregar con callbacks:

```python
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True
)

```

---

## 🚀 Conclusión

El notebook **“Overfitting and Underfitting”** cierra la introducción mostrando cómo lograr el balance ideal:

- **Underfitting** → modelo demasiado simple o poco entrenado.
- **Overfitting** → modelo demasiado complejo o entrenado de más.
- La clave está en usar técnicas de **regularización, early stopping y buen diseño de arquitectura** para alcanzar un punto de generalización óptimo.

---

## 🧠 **Mapa Conceptual – Introducción al Deep Learning**

### 1. **A Single Neuron**

- Entrada = **features** (xix_ixi).
- Parámetros = **pesos** (wiw_iwi) + **bias** (bbb).
- Salida = combinación lineal + **función de activación**.
- 👉 Base del deep learning.

---

### 2. **Deep Neural Networks**

- Varias **capas ocultas** = profundidad.
- Cada capa aprende **representaciones más abstractas**.
- Funciones de activación (ReLU, tanh, softmax).
- 👉 Redes profundas = más expresivas.

---

### 3. **Stochastic Gradient Descent (SGD)**

- Definimos una **función de pérdida** (MSE, cross-entropy).
- Ajustamos pesos con el gradiente:w←w−η⋅∇L
    
    w←w−η⋅∇Lw \leftarrow w - \eta \cdot \nabla L
    
- Modos:
    - **Batch** (todo el dataset).
    - **Stochastic** (1 ejemplo).
    - **Mini-batch** (subconjuntos).
- 👉 **Learning rate** controla el tamaño de los pasos.

---

### 4. **Binary Classification**

- Problema de salida **0 o 1**.
- Arquitectura:
    - Capa oculta → ReLU.
    - Capa de salida → **sigmoid**.
- Función de pérdida = **binary cross-entropy**.
- Métrica = **accuracy**.
- 👉 Salida = probabilidad.

---

### 5. **Dropout & Batch Normalization**

- **Dropout**: apaga neuronas al azar → previene overfitting.
- **BatchNorm**: normaliza activaciones por batch → acelera y estabiliza el entrenamiento.
- 👉 Ambas mejoran generalización y estabilidad.

---

### 6. **Overfitting & Underfitting**

- **Underfitting**: modelo demasiado simple / mal entrenado.
- **Overfitting**: modelo demasiado complejo / entrenado en exceso.
- Diagnóstico → **learning curves**.
- Soluciones:
    - Underfitting → más capas, más épocas.
    - Overfitting → más datos, dropout, regularización, early stopping.
- 👉 Buscar el **equilibrio**.

---

## 🔗 **Relaciones clave**

- La **neurona individual** es el ladrillo → se combinan en **deep nets**.
- El **aprendizaje** ocurre gracias a **SGD y pérdida**.
- Según el **tipo de problema** → usamos activaciones y pérdidas distintas.
- El entrenamiento necesita **regularización** (Dropout, BatchNorm) para evitar overfitting.
- Siempre hay que balancear entre **underfitting ↔ overfitting**.

---

📌 Este mapa une toda la serie en una progresión clara:

**Neurona → Red profunda → Entrenamiento (SGD) → Aplicaciones (Clasificación) → Regularización (Dropout/BatchNorm) → Generalización (Overfitting/Underfitting).**

# **Google Deep Learning:**

## [Neural Networks Course](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook)

## 📌 Resumen general

El **Deep Learning Tuning Playbook** responde a la pregunta:

👉 *“Tengo un modelo de deep learning, ¿cómo hago que funcione mejor?”*

Explica:

- Cómo **empezar simple**.
- Qué **hiperparámetros ajustar primero**.
- Cómo diagnosticar **overfitting y underfitting**.
- Qué técnicas usar para mejorar rendimiento y generalización.

---

## 📖 Conceptos principales desarrollados

### 1. **Empieza simple**

- Arrancar con un modelo pequeño y sencillo.
- Confirmar que funciona antes de aumentar complejidad.
- Esto evita perder tiempo con arquitecturas enormes mal configuradas.

---

### 2. **Capacidad del modelo**

- **Baja capacidad → underfitting.**
- **Alta capacidad → riesgo de overfitting.**
- Ajustar capacidad = cambiar número de capas, neuronas y parámetros.

👉 Consejo: comienza con una red pequeña y **escálala hasta que aparezca overfitting**.

---

### 3. **Diagnóstico: learning curves**

- Gráfico de **pérdida de entrenamiento vs validación**.
- Sirve para identificar:
    - Underfitting (ambas curvas altas).
    - Overfitting (divergencia entre entrenamiento y validación).
    - Buen ajuste (ambas bajas y cercanas).

---

### 4. **Regularización**

Técnicas para combatir el overfitting:

- **Dropout** → apaga neuronas al azar.
- **Weight decay (L2 regularization)** → penaliza pesos grandes.
- **Data augmentation** → crea ejemplos sintéticos a partir de datos reales (ej. rotar imágenes).
- **Early stopping** → detiene entrenamiento cuando la validación empeora.

---

### 5. **Batch Normalization**

- Normaliza activaciones capa por capa.
- Beneficios:
    - Estabiliza el entrenamiento.
    - Permite tasas de aprendizaje más grandes.
    - Puede actuar como regularizador.

---

### 6. **Learning Rate (tasa de aprendizaje)**

- Es el hiperparámetro más importante.
- Ajustarlo primero antes de cambiar arquitectura.
- Estrategias:
    - **Learning rate schedules** (decay progresivo).
    - **Warmup** (empezar bajo y subir).

---

### 7. **Batch Size**

- Tamaños chicos → entrenamiento más ruidoso, mejor generalización.
- Tamaños grandes → entrenamiento más estable, pero puede sobreajustar.

---

### 8. **Optimizadores**

- Recomendación general → usar **Adam** como punto de partida.
- Luego probar con SGD + momentum si se busca más control.

---

### 9. **Transfer Learning**

- Si el dataset es pequeño, usar un modelo ya pre-entrenado y ajustarlo (*fine-tuning*).
- Ahorra tiempo y mejora resultados.

---

### 10. **Hiperparámetros importantes a ajustar (en orden de prioridad)**

1. **Learning rate.**
2. **Tamaño de batch.**
3. **Arquitectura (capas y neuronas).**
4. **Regularización (dropout, weight decay).**
5. **Número de épocas y early stopping.**

---

### 11. **Proceso iterativo recomendado**

1. Empieza con un modelo pequeño.
2. Ajusta learning rate hasta que aprenda.
3. Aumenta capacidad del modelo hasta ver overfitting.
4. Aplica regularización para reducir overfitting.
5. Ajusta otros hiperparámetros.

👉 La idea es **iterar y diagnosticar con learning curves**, no cambiar todo a la vez.

---

## 🚀 Conclusión

El **Deep Learning Tuning Playbook de Google** es una guía práctica que enseña:

- **Cómo empezar** con modelos simples.
- **Qué hiperparámetros ajustar primero** (learning rate y batch size).
- **Cómo diagnosticar underfitting/overfitting** con curvas de entrenamiento.
- **Qué técnicas aplicar** (dropout, weight decay, batchnorm, data augmentation, early stopping).
- **Cómo iterar** para mejorar paso a paso.

👉 En resumen: es un **manual de buenas prácticas** para que las redes neuronales profundas sean entrenadas de manera eficiente y generalicen bien en problemas reales.

# **PyTorch Lightning:**

## [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)

## [Getting Started Guide](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

# 📌 ¿Qué es PyTorch Lightning?

Es un **framework de alto nivel** construido sobre PyTorch que:

- **Simplifica** el entrenamiento de redes neuronales.
- **Separa** la lógica del modelo (qué es la red) de la infraestructura (cómo se entrena, en cuántas GPUs, logging, etc.).
- Permite escribir **menos código repetitivo** y enfocarse en el modelo.

👉 Piensa en Lightning como una forma de “organizar tu código PyTorch en limpio y escalable”.

---

# 📖 Conceptos principales de la documentación

### 1. **LightningModule**

Es el bloque central. Contiene:

- `__init__`: definición de capas y modelo.
- `forward`: cómo pasan los datos por la red.
- `training_step`: qué ocurre en cada batch de entrenamiento (predicción, pérdida).
- `validation_step` / `test_step`: lógica para validación y test.
- `configure_optimizers`: qué optimizador usar (Adam, SGD, etc.).

👉 Esto reemplaza al entrenamiento manual con `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.

---

### 2. **Trainer**

Es el objeto que **orquesta el entrenamiento**.

Ejemplo:

```python
trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=1)
trainer.fit(model, train_dataloader, val_dataloader)

```

- `max_epochs=5`: número de épocas.
- `accelerator="gpu"`: usa GPU automáticamente si hay.
- `devices=1`: cuántas GPUs usar.
- `fit`: entrena el modelo con los datos.

👉 El `Trainer` maneja todo lo repetitivo: bucles de entrenamiento, validación, callbacks, logging, checkpoints.

---

### 3. **Callbacks**

Permiten personalizar entrenamientos sin ensuciar el código:

- **EarlyStopping** → detener si la validación no mejora.
- **ModelCheckpoint** → guardar el mejor modelo.
- **LearningRateMonitor** → registrar cómo cambia el LR.

---

### 4. **DataModules**

Organizan los **datasets y dataloaders** en un solo bloque.

Incluyen:

- `prepare_data()`: descarga/prepara dataset.
- `setup()`: divide en train/val/test.
- `train_dataloader()`, `val_dataloader()`, `test_dataloader()`.

👉 Facilita el reuso y orden del código.

---

### 5. **Escalabilidad**

- Entrenar en **múltiples GPUs** sin cambiar código.
- Entrenamiento **distribuido en clústeres**.
- Soporte para **TPUs**.

---

### 6. **Integraciones**

Lightning se integra con:

- **Loggers** → TensorBoard, WandB, MLFlow.
- **Plugins** → mixed precision (fp16), pruning, cuantización.
- **HuggingFace, TorchMetrics, Optuna**.

---

# 📖 Getting Started Guide (Introducción)

Ejemplo mínimo de un modelo de clasificación con Lightning:

```python
import pytorch_lightning as pl
from torch import nn, optim
import torch

class LitClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(28 * 28, 10)  # ejemplo MNIST

    def forward(self, x):
        return torch.relu(self.layer(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

# Entrenamiento
trainer = pl.Trainer(max_epochs=5)
model = LitClassifier()
trainer.fit(model, train_dataloader, val_dataloader)

```

👉 Con pocas líneas se define el modelo, el loop de entrenamiento y validación. El resto lo maneja Lightning.

---

# 🚀 Conclusión

Los links de **PyTorch Lightning** enseñan:

- Cómo organizar modelos con `LightningModule`.
- Cómo entrenarlos fácilmente con `Trainer`.
- Cómo añadir callbacks, logging y escalabilidad sin cambiar la lógica.
- Cómo estructurar datasets con `DataModule`.
- Cómo empezar con un ejemplo práctico y simple (MNIST).

👉 En resumen: **Lightning hace que PyTorch sea más limpio, menos repetitivo y más escalable.**