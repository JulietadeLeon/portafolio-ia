---
title: "Lecturas"
date: 2025-01-01
---

## **Aprendizajes al completar UT4:**

- Comprender la arquitectura Transformer y los mecanismos de atención.
- Implementar técnicas de fine-tuning de modelos de lenguaje (LoRA, QLoRA).
- Desarrollar pipelines RAG (Retrieval-Augmented Generation) con LangChain.
- Crear chatbots multi-turn con gestión de memoria usando LangGraph.
- Aplicar técnicas de prompt engineering y evaluación de diálogos.
- Integrar modelos de Hugging Face en aplicaciones organizacionales.
- Implementar sistemas de QA con vector stores (FAISS, Pinecone).

## Lecturas previas obligatorias

- [A Complete Guide to Natural Language Processing](https://www.deeplearning.ai/resources/natural-language-processing/)
- [Introduction to Text Classification - Google](https://developers.google.com/machine-learning/guides/text-classification)
- [Hugging Face LLM Course – Capítulo 1: Introducción a los Transformers, encoders/decoders, atención](https://huggingface.co/learn/llm-course/es/chapter1/1?utm_source=chatgpt.com)
- [Hugging Face Agents Course Unit 1 Introduction to agents](https://huggingface.co/learn/agents-course/unit0/introduction)

# Procesamiento del Lenguaje Natural y Modelos de Lenguaje

## **1. A Complete Guide to Natural Language Processing (DeepLearning.AI)**

El artículo **“A Complete Guide to Natural Language Processing”** de *DeepLearning.AI* ofrece una visión integral sobre el campo del **Procesamiento del Lenguaje Natural (NLP)**, una rama de la inteligencia artificial que tiene como objetivo permitir que las máquinas comprendan, interpreten y generen lenguaje humano.

El texto comienza destacando la evolución histórica del NLP, desde los primeros sistemas basados en reglas gramaticales hasta los enfoques actuales impulsados por el aprendizaje profundo. En sus inicios, los sistemas lingüísticos dependían de estructuras predefinidas y limitadas, lo que restringía su capacidad para manejar la ambigüedad y variabilidad del lenguaje natural. Con la llegada del *machine learning* y las redes neuronales, el campo experimentó un cambio radical: los modelos comenzaron a **aprender representaciones estadísticas del lenguaje a partir de grandes volúmenes de datos textuales**.

Entre los principales **componentes del pipeline de NLP**, la lectura detalla:

- **Tokenización:** proceso de segmentar un texto en unidades mínimas (palabras, subpalabras o caracteres), facilitando el análisis posterior.
- **Lematización y stemming:** reducción de las palabras a su raíz o forma base, eliminando variaciones morfológicas.
- **Vectorización:** transformación de las palabras en representaciones numéricas (embeddings), que capturan relaciones semánticas y contextuales.
- **Modelado secuencial:** uso de arquitecturas neuronales como RNNs, LSTMs o Transformers para modelar la dependencia entre palabras en una oración.

El documento también profundiza en las **aplicaciones prácticas** del NLP: clasificación de texto, análisis de sentimientos, traducción automática, generación de texto y sistemas conversacionales. Asimismo, resalta los **desafíos éticos y técnicos**, como la presencia de sesgos en los datos, la dificultad para interpretar modelos complejos y la necesidad de garantizar transparencia en los sistemas lingüísticos automatizados.

En síntesis, la guía de DeepLearning.AI plantea al NLP como una disciplina interdisciplinaria que combina lingüística, estadística y deep learning para acercar la comunicación humana al lenguaje computacional, convirtiéndose en una de las áreas más transformadoras de la inteligencia artificial moderna.

---

## **2. Introduction to Text Classification (Google Machine Learning Guides)**

La lectura de Google **“Introduction to Text Classification”** presenta una descripción técnica y conceptual de la **clasificación de texto**, una de las tareas más comunes dentro del procesamiento del lenguaje natural. Esta técnica consiste en asignar automáticamente categorías o etiquetas a fragmentos de texto, como correos electrónicos, reseñas o noticias, en función de su contenido semántico.

El artículo comienza explicando que el texto, al ser un dato no estructurado, necesita pasar por un proceso de **preprocesamiento y vectorización** antes de poder ser analizado por un algoritmo. Entre los pasos iniciales se encuentran la eliminación de palabras irrelevantes (*stopwords*), la normalización, la tokenización y la representación del texto en formato numérico.

Se introducen distintos **métodos de representación**:

- **Bag of Words (BoW):** donde cada documento se representa como una bolsa de palabras sin considerar el orden, basándose solo en la frecuencia de aparición.
- **TF-IDF (Term Frequency–Inverse Document Frequency):** que pondera la importancia de una palabra en un documento según su frecuencia local y global en el corpus.
- **Embeddings:** representaciones densas que capturan relaciones semánticas, como Word2Vec o GloVe, y posteriormente las más avanzadas como las obtenidas con modelos basados en Transformers.

La lectura también describe los **modelos de clasificación supervisada**, como la regresión logística, las máquinas de soporte vectorial (SVM), las redes neuronales y los modelos basados en deep learning.

El proceso de entrenamiento implica disponer de un conjunto de textos etiquetados, a partir del cual el modelo aprende a identificar patrones lingüísticos y semánticos que caracterizan cada clase.

En la etapa de evaluación, se introducen métricas esenciales como la **precisión (accuracy)**, la **recuperación (recall)** y la **F1-score**, que permiten medir el desempeño de los modelos de manera equilibrada. Google destaca también la importancia del manejo de **desbalance de clases**, el uso de **regularización** para evitar el sobreajuste y la interpretación responsable de los resultados, especialmente en aplicaciones sensibles (por ejemplo, moderación de contenido o clasificación de opiniones).

La lectura concluye subrayando que la clasificación de texto es un punto de partida fundamental para comprender tareas más complejas de NLP, y que su efectividad depende no solo del modelo utilizado, sino también de la calidad y representatividad de los datos de entrenamiento.

---

## **3. Hugging Face LLM Course – Capítulo 1: Introducción a los Transformers, Encoders/Decoders y Atención**

El **Capítulo 1 del curso de LLMs de Hugging Face** introduce la arquitectura **Transformer**, que revolucionó el campo del procesamiento del lenguaje natural y sentó las bases de los modelos de lenguaje grandes (LLMs) actuales, como GPT, BERT o T5.

El texto comienza explicando el cambio de paradigma que trajo esta arquitectura frente a los modelos secuenciales tradicionales como las RNN y LSTM. Mientras que estos últimos procesaban la información palabra por palabra en orden, los Transformers emplean un mecanismo llamado **atención (attention mechanism)**, que les permite analizar simultáneamente todas las palabras de una oración y evaluar la relevancia contextual entre ellas.

Este principio se formaliza mediante la **self-attention**, donde cada palabra “atiende” a las demás dentro de la misma secuencia, ponderando su influencia en la representación final. De este modo, el modelo puede capturar dependencias a largo plazo, comprender el contexto global y manejar oraciones más extensas sin las limitaciones de memoria típicas de los modelos recurrentes.

La arquitectura Transformer se estructura en **dos componentes principales: encoder y decoder**:

- El **encoder** toma una secuencia de entrada (por ejemplo, una oración en inglés) y genera una representación interna o embedding contextualizado.
- El **decoder** utiliza esa representación para producir una salida (por ejemplo, la traducción al español), procesando la información paso a paso con atención tanto al contexto de entrada como a las palabras ya generadas.

Esta organización encoder-decoder hace posible una amplia variedad de tareas: traducción automática, resumen de texto, respuesta a preguntas o generación de lenguaje natural.

El curso además introduce las nociones de **preentrenamiento y fine-tuning** en grandes volúmenes de datos. Los modelos Transformer se entrenan inicialmente para aprender las estructuras del lenguaje general (por ejemplo, prediciendo palabras faltantes) y luego se ajustan para tareas específicas mediante fine-tuning.

Finalmente, Hugging Face subraya la importancia del **aprendizaje transferido** en los LLMs, la escalabilidad del entrenamiento y el rol de los embeddings en la comprensión semántica del lenguaje. Esta lectura permite comprender por qué los Transformers se convirtieron en el pilar tecnológico de la inteligencia artificial generativa moderna.

---

## **4. Hugging Face Agents Course – Unit 1: Introduction to Agents**

La **Unidad 1 del curso de Hugging Face sobre Agents** introduce un nuevo nivel de abstracción dentro de la inteligencia artificial: los **agentes autónomos**.

Un agente puede definirse como un sistema que **percibe su entorno, razona sobre él y ejecuta acciones de forma autónoma para alcanzar un objetivo**, interactuando con modelos de lenguaje, APIs y herramientas externas.

El texto contextualiza esta idea dentro del auge de los **LLMs como motores cognitivos**, capaces de interpretar instrucciones en lenguaje natural y decidir cómo actuar. Sin embargo, mientras un modelo de lenguaje genera texto o predicciones, un agente va un paso más allá: **combina razonamiento, planificación y ejecución de tareas** en entornos dinámicos.

Se explican los **componentes principales de un agente**:

- **Modelo base (LLM):** encargado de interpretar el lenguaje y generar razonamientos.
- **Herramientas externas (tools):** funciones o APIs que el agente puede invocar (por ejemplo, hacer una búsqueda, enviar un correo o ejecutar código).
- **Memoria:** almacena contexto y resultados previos, permitiendo que el agente aprenda de interacciones pasadas.
- **Política o plan de acción:** define cómo el agente decide qué hacer a continuación, basándose en objetivos y retroalimentación.

El curso destaca que los agentes pueden ser **reactivos** (responden a eventos) o **proactivos** (toman la iniciativa para lograr metas), y que su desarrollo abre la puerta a sistemas más inteligentes y autónomos, capaces de integrar razonamiento simbólico, lenguaje natural y ejecución práctica.

Asimismo, se subraya el papel de la **seguridad, trazabilidad y control ético** en el diseño de agentes, ya que su autonomía implica riesgos potenciales si no se definen correctamente los límites de sus acciones o las fuentes de información.

El material enfatiza la importancia de los frameworks abiertos, como los proporcionados por Hugging Face, que permiten a los desarrolladores construir agentes personalizados conectados a modelos, herramientas y bases de conocimiento diversas.

En suma, esta lectura introduce la evolución natural del campo del NLP y los LLMs hacia sistemas **interactivos, dinámicos y orientados a tareas**, donde los modelos de lenguaje dejan de ser pasivos y comienzan a actuar como entidades cognitivas que colaboran con los humanos en la resolución de problemas reales.