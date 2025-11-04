---
title: "Apuntes"
date: 2025-01-01
---

## **Clase 7 – Fundamentos de Computer Vision y Redes Convolucionales**

La clase introduce los principios biológicos y computacionales que inspiran la **visión por computadora**, tomando como punto de partida el estudio de la **corteza visual humana** realizado por *Hubel y Wiesel (1959–1962)*. Estos investigadores demostraron que el procesamiento visual se organiza jerárquicamente, donde distintas neuronas responden a patrones visuales de creciente complejidad (líneas, formas, movimiento, etc.). Este enfoque biológico inspiró el diseño de las **Redes Neuronales Convolucionales (CNNs)**.

Se diferencia entre **datos estructurados** (como tablas numéricas) y **datos no estructurados** (imágenes, audio o texto), destacando que las CNNs permiten el *representation learning*, es decir, el aprendizaje automático de las características relevantes sin intervención manual.

A nivel técnico, se analiza la **operación de convolución**, que utiliza filtros o *kernels* para detectar características locales de una imagen (bordes, esquinas, texturas). A medida que se avanza en profundidad dentro de la red, las capas convolucionales aprenden representaciones cada vez más complejas y abstractas.

Se discuten además los **hiperparámetros clave** de las CNNs:

- Número y tamaño de filtros,
- *Stride* (desplazamiento),
- *Padding* (relleno de bordes).

La clase también aborda las **capas de pooling**, en particular el *max pooling*, que resume la información de regiones pequeñas reduciendo la dimensionalidad y aumentando la invarianza espacial del modelo.

Finalmente, se introduce el **Transfer Learning**, técnica que permite reutilizar modelos preentrenados (por ejemplo, en grandes datasets como ImageNet) para resolver nuevas tareas con menor costo computacional. Se diferencian dos enfoques:

- **Feature Extraction:** se utilizan los pesos preentrenados como extractor de características y se entrena solo la capa final de clasificación.
- **Fine-Tuning:** se reentrenan las últimas capas del modelo base junto con las nuevas capas, ajustando parcialmente los pesos previos.

Esta clase sienta las bases conceptuales del aprendizaje profundo aplicado a imágenes, estableciendo la conexión entre la neurociencia, la matemática de las convoluciones y la práctica moderna de redes neuronales.

---

## **Clase 8 – Data Augmentation e Inteligencia Artificial Explicable**

La octava clase aborda dos ejes fundamentales en la práctica moderna de la visión por computadora: el **aumento de datos (Data Augmentation)** y la **interpretabilidad de modelos**.

El *data augmentation* consiste en aplicar transformaciones aleatorias a las imágenes del conjunto de entrenamiento (rotaciones, traslaciones, reflejos, ajustes de brillo o zoom) con el fin de **incrementar la diversidad de los datos** y **evitar el sobreajuste** (*overfitting*). Esto permite que el modelo aprenda a reconocer patrones bajo distintas condiciones y sea más robusto frente a variaciones reales como cambios de luz, posición o perspectiva.

Además, se contrasta el *data augmentation* con la **generación de datos sintéticos**, remarcando que mientras el primero parte de ejemplos reales y aplica modificaciones locales, el segundo crea imágenes desde cero mediante redes generativas (como las GANs). Si bien los datos sintéticos ofrecen control semántico, suelen tener mayor costo computacional y riesgos de falta de realismo.

En la segunda parte, la clase se centra en la **IA Explicable (Explainable AI)**, cuya finalidad es comprender cómo y por qué los modelos toman determinadas decisiones. Se busca reducir la naturaleza de “caja negra” de las redes profundas, mejorando la transparencia, la confianza y la capacidad de *debug*.

Se presentan dos métodos de interpretación visual:

- **Integrated Gradients:** mide qué píxeles contribuyen más a la salida correcta del modelo, asignando puntajes de importancia.
- **GradCAM (Gradient-weighted Class Activation Mapping):** analiza las activaciones de la última capa convolucional para mostrar qué regiones de la imagen influyeron en la clasificación.

Estas técnicas permiten visualizar las “zonas de atención” del modelo, facilitando la comprensión de errores y la evaluación de si el modelo está enfocándose en los aspectos correctos de la imagen.

---

## **Clase 9 – Detección de Objetos y Métricas de Evaluación**

La novena clase introduce los **problemas avanzados de visión por computadora**, centrados en la **detección de objetos (Object Detection)** y sus diferencias con la **clasificación de imágenes (Image Classification)**.

En la **clasificación de imágenes**, el modelo recibe una imagen completa y predice una sola clase global (“contiene un perro”). En cambio, la **detección de objetos** combina dos tareas simultáneas:

1. **Localización:** determinar *dónde* está el objeto dentro de la imagen, mediante *bounding boxes* (coordenadas [x₁, y₁, x₂, y₂]);
2. **Clasificación:** identificar *qué* es el objeto (persona, vehículo, animal, etc.), junto con una medida de **confianza** o *score*.

Se analizan las **arquitecturas más representativas** de este tipo de modelos:

- **Two-Stage Detectors (R-CNN, Fast R-CNN, Faster R-CNN):** primero proponen regiones de interés y luego las clasifican individualmente. Faster R-CNN introduce una *Region Proposal Network (RPN)* que aprende automáticamente a generar regiones relevantes.
- **Single-Stage Detectors (YOLO – You Only Look Once):** procesan toda la imagen en una sola pasada, dividiéndola en una cuadrícula y prediciendo directamente las coordenadas, tamaños, clases y niveles de confianza. Esta arquitectura es más rápida y eficiente para aplicaciones en tiempo real.

Para refinar las detecciones, se aplica la técnica **Non-Maximum Suppression (NMS)**, que elimina cajas redundantes conservando solo las más confiables mediante la métrica **IoU (Intersection over Union)**, que mide la superposición entre predicción y realidad.

Finalmente, se explican las **métricas de evaluación** utilizadas en detección:

- **Precision** y **Recall**, que miden exactitud y cobertura;
- **Average Precision (AP)** y **mean Average Precision (mAP)**, que promedian los resultados sobre todas las clases y distintos umbrales de IoU.
    
    Un mAP de 0.7 se considera un buen modelo, mientras que valores cercanos a 0.9 reflejan un rendimiento sobresaliente.
    

Esta clase proporciona el marco conceptual y técnico para abordar tareas de detección complejas, fundamentales en aplicaciones como conducción autónoma, análisis de video o monitoreo de seguridad.

---

## **Clase 10 – Segmentación de Imágenes y Zero-Shot Learning**

La décima clase profundiza en el concepto de **segmentación**, tarea que busca identificar píxel a píxel las regiones de una imagen que pertenecen a un mismo objeto o clase.

Se distinguen tres tipos principales:

- **Semantic Segmentation:** asigna la misma etiqueta a todos los píxeles de un mismo tipo de objeto (por ejemplo, “todos los autos” en un color uniforme).
- **Instance Segmentation:** identifica cada instancia individual de un objeto, incluso si son de la misma clase.
- **Panoptic Segmentation:** combina ambos enfoques, representando tanto clases como instancias de forma simultánea.

Se presenta el modelo **SAM (Segment Anything Model)**, desarrollado como un sistema **promptable, generalista e interactivo**. SAM puede recibir distintos tipos de *prompts* o instrucciones visuales, como puntos, cajas o máscaras, e incluso texto en su versión SAM-2. Su fortaleza reside en haber sido entrenado con **billones de máscaras**, lo que le permite funcionar en modo **zero-shot**, es decir, segmentar objetos sin necesidad de ser reentrenado en cada dominio nuevo.

El concepto de **Zero-Shot Learning** refiere a la capacidad de un modelo para reconocer o segmentar clases nunca vistas durante el entrenamiento, mediante la **transferencia de representaciones generales** aprendidas previamente. SAM utiliza *embeddings* visuales que capturan similitudes semánticas y pueden razonar por analogía (“esto se parece a aquello”).

También se describe la **métrica Dice**, utilizada para cuantificar el grado de superposición entre la máscara predicha y la real (*ground truth*). Cuanto mayor es el coeficiente de Dice, mayor es la precisión de la segmentación.

Por último, se analiza cuándo conviene aplicar **fine-tuning** sobre modelos como SAM: es recomendable cuando se trabaja en dominios específicos (como imágenes médicas o industriales) o con objetos pequeños y detallados. En cambio, el modo zero-shot es suficiente para imágenes naturales o cuando se dispone de pocos recursos.

Esta clase consolida los conocimientos sobre las tareas más avanzadas de visión por computadora, mostrando la evolución desde la clasificación simple hasta la segmentación inteligente impulsada por modelos generativos y multimodales.
