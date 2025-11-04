---
title: "Lecturas"
date: 2025-01-01
---

# Aprendizajes al completar UT3: 

• Comprender la arquitectura de redes convolucionales (CNNs) y su motivación

• Implementar transfer learning con modelos pre-entrenados

• Aplicar técnicas avanzadas de data augmentation para robustez

• Desarrollar sistemas de detección de objetos con YOLO para casos reales

• Implementar segmentación con SAM

# Lecturas

### **Lecturas minimas:**

[Kaggle Computer Vision (Completo) https://www.kaggle.com/learn/computer-vision](https://www.kaggle.com/learn/computer-vision):

- The Convolutional Classifier
- Convolution and ReLU
- Maximum Pooling
- The Sliding Window
- Custom Convnets
- Data Augmentation

HuggingFace Computer Vision:

- [Computer Vision Course - Unit 1: Fundamentals https://huggingface.co/learn/computer-vision-course/unit1/chapter1/motivation](https://huggingface.co/learn/computer-vision-course/unit1/chapter1/motivation)
- [Computer Vision Course - Unit 2: CNNs https://huggingface.co/learn/computer-vision-course/unit2/cnns/introduction](https://huggingface.co/learn/computer-vision-course/unit2/cnns/introduction)
- --------------------------------------------------------------------------------------------

### **Lecturas totales:**

Clase 7 - CNNs + Transfer Learning:

- Kaggle CV - The Convolutional Classifier
- Kaggle CV - Convolution and ReLU
- timm Documentation - Model Overview

Clase 8 - Data Augmentation:

- Kaggle CV - Data Augmentation
- Kaggle CV - Transfer Learning
- Albumentations Documentation

Clase 9 - Object Detection:

- HuggingFace CV Course - Object Detection https://huggingface.co/learn/computer-vision-course/unit6/basic-cv-tasks/object_detection
- Kaggle CV - Custom ConvNets
- YOLOv8 Documentation
- COCO Dataset Guide

Clase 10 - Segmentation:

- HuggingFace CV Course - Segmentation https://huggingface.co/learn/computer-vision-course/unit6/basic-cv-tasks/segmentation
- Kaggle CV - Maximum Pooling
- SAM 2 Documentation

LECTURAS ADICIONALES COMPLEMENTARIAS:

- Deep Learning for Computer Vision (Stanford CS231n) https://cs231n.github.io/
- --------------------------------------------------------------------------------------------

### **Herramientas:**

**Deep Learning Frameworks:**

- **PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/**
- **timm (SOTA models): https://timm.fast.ai/**
- **torchvision: https://pytorch.org/vision/stable/index.html**

**Object Detection:**

- **YOLO (Ultralytics): https://docs.ultralytics.com/**
- **COCO API: https://github.com/cocodataset/cocoapi**
- **detectron2: https://detectron2.readthedocs.io/**

**Segmentation:**

- **SAM (Meta AI): https://github.com/facebookresearch/segment-anything-2**
- **HuggingFace Transformers: https://huggingface.co/docs/transformers/**

**Data Augmentation:**

- **Albumentations: https://albumentations.ai/docs/**

---

# **Fundamentos y Técnicas de Visión por Computadora**

## **1. The Convolutional Classifier**

El clasificador convolucional constituye la arquitectura central de las redes neuronales aplicadas a visión por computadora. Se basa en la idea de utilizar **capas convolucionales** para extraer características espaciales de las imágenes, como bordes, texturas o patrones de forma, preservando su estructura bidimensional.

A diferencia de los modelos densos tradicionales, las convoluciones permiten **reducir la cantidad de parámetros** y **capturar dependencias locales** entre píxeles cercanos.

Un clasificador convolucional típico combina una secuencia de capas convolucionales y de pooling para generar mapas de características, los cuales luego son aplanados y conectados a **capas densas (fully connected)** que realizan la clasificación final.

Este enfoque ha demostrado ser altamente eficiente para tareas como el reconocimiento de objetos, la segmentación semántica o la detección facial, ya que logra aprender automáticamente representaciones jerárquicas de las imágenes, donde las primeras capas detectan patrones simples (bordes o colores) y las últimas combinan esas características en estructuras más complejas.

---

## **2. Convolution and ReLU**

Las **convoluciones** constituyen el proceso fundamental mediante el cual una red neuronal convolucional aprende a reconocer patrones visuales. Matemáticamente, la convolución consiste en aplicar un **filtro o kernel** que se desliza sobre la imagen, multiplicando los valores de los píxeles por los pesos del filtro y sumando los resultados. Cada filtro detecta un tipo particular de característica visual, como líneas horizontales, diagonales o texturas.

Posteriormente, se aplica una función de activación no lineal, siendo la **ReLU (Rectified Linear Unit)** la más utilizada. Su función es transformar los valores negativos en cero, permitiendo que el modelo aprenda representaciones no lineales y evitando el problema de desvanecimiento del gradiente.

El uso conjunto de la convolución y ReLU permite que el modelo conserve las características relevantes de la imagen, elimine el ruido y mantenga la capacidad de aprendizaje profundo a través de múltiples capas sucesivas.

---

## **3. Maximum Pooling**

El **max pooling** o “agrupamiento máximo” es una técnica que reduce la dimensionalidad de los mapas de características, manteniendo la información más relevante.

Consiste en dividir el mapa de características en regiones (por ejemplo, de 2x2 píxeles) y seleccionar el valor máximo de cada región. De esta manera, se consigue una **reducción en el número de parámetros y en el costo computacional**, al mismo tiempo que se obtiene **invarianza espacial**, es decir, la capacidad de reconocer un objeto aunque se encuentre ligeramente desplazado en la imagen.

Este procedimiento contribuye a la robustez del modelo y ayuda a prevenir el sobreajuste, al concentrar la información esencial y descartar detalles irrelevantes o redundantes.

---

## **4. The Sliding Window**

El concepto de **ventana deslizante** surge como una técnica para aplicar un mismo modelo o filtro a diferentes regiones de una imagen. En el contexto de redes convolucionales, permite **escanear la imagen completa** sin perder información espacial.

En aplicaciones clásicas de visión por computadora, como la detección de objetos, la ventana deslizante se utilizaba para recorrer la imagen en distintas posiciones y tamaños, aplicando un clasificador en cada región.

En las redes convolucionales modernas, esta idea se implementa de manera intrínseca: las convoluciones actúan como múltiples ventanas deslizantes que aprenden a detectar patrones relevantes en diferentes ubicaciones.

Este principio es esencial para que los modelos de visión artificial reconozcan objetos independientemente de su posición dentro del campo visual, lo cual resulta crítico en tareas de clasificación y detección.

---

## **5. Custom Convnets**

Los **custom convnets** hacen referencia al diseño y entrenamiento de redes convolucionales personalizadas, adaptadas a un problema o conjunto de datos específico.

En lugar de utilizar arquitecturas predefinidas (como VGG, ResNet o MobileNet), se pueden construir modelos desde cero, definiendo la cantidad de capas, filtros, funciones de activación, y otros hiperparámetros.

El objetivo es lograr un equilibrio entre **complejidad y capacidad de generalización**, ajustando la profundidad y el tamaño de los filtros según las características del dataset.

La creación de convnets personalizadas permite experimentar con distintas configuraciones y optimizaciones, como regularización (Dropout, Batch Normalization), inicialización de pesos o tipos de pooling, y analizar su impacto en el rendimiento del modelo. Este enfoque fomenta la comprensión profunda del funcionamiento interno de las redes neuronales convolucionales y la capacidad de adaptarlas a contextos de aplicación reales.

---

## **6. Data Augmentation**

El **aumento de datos (Data Augmentation)** es una técnica esencial para mejorar la capacidad de generalización de los modelos de visión artificial. Consiste en generar nuevas versiones de las imágenes de entrenamiento aplicando **transformaciones aleatorias** como rotaciones, traslaciones, recortes, cambios de brillo o volteos horizontales.

Estas modificaciones no alteran el contenido semántico de la imagen, pero sí introducen variabilidad, simulando condiciones de iluminación o perspectiva diferentes.

Al aumentar artificialmente el tamaño y la diversidad del conjunto de entrenamiento, el modelo aprende a reconocer patrones de manera más robusta y reduce el riesgo de sobreajuste.

En el contexto del aprendizaje profundo, las bibliotecas como TensorFlow y PyTorch permiten aplicar estas transformaciones en tiempo real durante el entrenamiento, integrando el aumento de datos en el pipeline de entrenamiento del modelo.

En síntesis, el data augmentation contribuye a la creación de modelos más estables, capaces de enfrentar imágenes reales con variaciones naturales en su entorno.

---

## **7. Computer Vision Fundamentals (HuggingFace - Unit 1)**

La **Unidad 1 del curso de HuggingFace sobre Visión por Computadora** aborda los fundamentos conceptuales de esta disciplina, explorando cómo los modelos de aprendizaje profundo permiten que las máquinas “vean” y comprendan el contenido de las imágenes.

El curso introduce las principales **tareas de visión artificial**, entre ellas la clasificación de imágenes, la detección de objetos, la segmentación semántica y la generación de imágenes.

Además, se discuten los desafíos técnicos y éticos asociados, como la necesidad de grandes volúmenes de datos, el sesgo en los conjuntos de entrenamiento y el impacto de la interpretación automática de imágenes en distintos campos (salud, seguridad, arte, etc.).

También se analizan los avances históricos en el área, desde los primeros métodos basados en extracción manual de características (como SIFT o HOG) hasta los actuales enfoques basados en **redes neuronales profundas**, que aprenden representaciones jerárquicas de las imágenes sin intervención humana.

Esta unidad proporciona el marco conceptual necesario para comprender el papel de las CNN en el procesamiento visual y cómo se integran en el ecosistema más amplio de la inteligencia artificial moderna.

---

## **8. Convolutional Neural Networks (HuggingFace - Unit 2)**

La **Unidad 2** profundiza en la arquitectura de las **Redes Neuronales Convolucionales (CNN)**, explicando su estructura, funcionamiento y aplicaciones prácticas.

Las CNN se caracterizan por su capacidad para aprender automáticamente los filtros y características relevantes de las imágenes a través de un proceso de entrenamiento supervisado. Se analizan los componentes principales:

- **Capas convolucionales:** responsables de extraer patrones locales.
- **Capas de activación:** como ReLU, que introducen no linealidad.
- **Capas de pooling:** que reducen la dimensionalidad manteniendo las características esenciales.
- **Capas completamente conectadas:** que combinan la información y producen las predicciones finales.

El curso también aborda temas como la **inicialización de pesos**, el **entrenamiento mediante backpropagation**, la **función de pérdida** y el uso de optimizadores (Adam, SGD, RMSprop). Además, se presentan ejemplos de arquitecturas modernas y su evolución, desde LeNet-5 hasta modelos más recientes y eficientes como **ResNet o EfficientNet**.

Finalmente, se destacan las ventajas del uso de **transfer learning** y **fine-tuning**, estrategias que permiten reutilizar modelos preentrenados para tareas específicas con menores costos computacionales y mejor rendimiento, especialmente en contextos con datos limitados.