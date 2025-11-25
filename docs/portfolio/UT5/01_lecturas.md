---
title: "Lecturas"
date: 2025-01-01
---

# Lecturas

### **Objetivo de la Unidad**

Al completar UT5, el estudiante será capaz de:

- Implementar pipelines de MLOps para deployment escalable.
- Aplicar mejores prácticas de monitoreo y mantenimiento de modelos.

### **Lecturas previas obligatorias (Evaluación el 25/11)**

1. **Production ML Systems** – módulo de *Real-world ML* en el Google Machine Learning Crash Course
    
    [https://developers.google.com/machine-learning/crash-course/production-ml-systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems)
    
    ## 1. Introducción: ¿Qué es un sistema de ML en producción?
    
    El módulo comienza subrayando que en entornos reales de negocio, el modelo de machine learning no es el centro de todo: constituye apenas una parte del sistema completo. Según Google, el código del modelo a menudo representa **menos del 5 %** del volumen total de código de un sistema ML en producción. [Google for Developers+1](https://developers.google.com/machine-learning/crash-course/production-ml-systems?utm_source=chatgpt.com)
    
    Se introduce una arquitectura genérica que comprende múltiples componentes: recolección de datos, verificación de datos, extracción de características, configuración, gestión de recursos de cómputo, infraestructuras de inferencia, monitorización y más. [Google for Developers](https://developers.google.com/machine-learning/crash-course/production-ml-systems?utm_source=chatgpt.com)
    
    El enfoque del módulo es preparar al profesional para diseñar, desplegar y mantener estos sistemas más allá del entrenamiento del modelo.
    
    ---
    
    ## 2. Paradigmas clave: Entrenamiento e inferencia – estático vs dinámico
    
    ### 2.1 Entrenamiento estático (Static Training)
    
    En este paradigma, el modelo se entrena una vez con un conjunto de datos fijado, y luego ese modelo se despliega. No se vuelve a entrenar hasta que se decida explícitamente una actualización.
    
    Ventajas: simplicidad, menos complejidad operativa.
    
    Desventajas: puede volverse obsoleto si los datos cambian o emergen nuevas condiciones.
    
    ### 2.2 Entrenamiento dinámico (Dynamic Training)
    
    Aquí el modelo se actualiza de forma recurrente —por ejemplo diaria o en streaming— en función de nuevos datos o cambios en la distribución. Esto permite adaptarse al “drift” de datos.
    
    El módulo explica cómo determinar cuál paradigma aplicar, según el dominio, la criticidad del servicio, el coste de actualización y la tasa de cambio de los datos.
    
    ### 2.3 Inferencia estática vs inferencia dinámica
    
    - **Inferencia estática**: el servicio de inferencia usa un modelo fijo, sin cambiar durante su periodo de vida.
    - **Inferencia dinámica**: el sistema puede cambiar sobre la marcha, gestionar versiones múltiples, o incluso adaptarse en tiempo real al contexto del usuario.
    
    El módulo destaca que la decisión entre estático y dinámico afecta la arquitectura, el pipeline de datos, los requisitos de monitorización y la lógica de despliegue.
    
    ---
    
    ## 3. ¿Cuándo transformar los datos?
    
    Una sección concisa aborda el lugar adecuado para realizar transformaciones de datos (feature engineering, normalización, limpieza).
    
    El módulo recomienda tener en cuenta:
    
    - Si la transformación debe ejecutarse en tiempo de entrenamiento, inferencia o ambos.
    - Evitar divergencias entre “serving-time” y “training-time” (skew).
    - Registrar todas las transformaciones como parte del pipeline para asegurar reproducibilidad.
    
    Este punto es critico para evitar errores de producción como el “training-serving skew”.
    
    ---
    
    ## 4. Testing del despliegue (‘Deployment Testing’)
    
    Antes de poner un modelo en producción, el módulo insiste en realizar pruebas automáticas:
    
    - Verificar que el servicio de inferencia responde correctamente bajo carga.
    - Validar que el pipeline de datos, el preprocesamiento y el modelo funcionan end-to-end.
    - Pruebas de regresión para asegurar que nuevas versiones no degradan desempeño.
    - Validación de contratos de datos: formatos, tamaños, rangos esperados.
    
    Este bloque resalta que el despliegue de un ML system no puede tratarse como un simple “push” de modelo, sino que requiere ingeniería robusta de producción.
    
    ---
    
    ## 5. Monitorización de los pipelines en producción
    
    Una de las partes más extensas del módulo (~25 min) se dedica a la monitorización operativa del pipeline ML:
    
    - Métricas clave: latencia de inferencia, tasa de error, frecuencia de fallo, distribución de predicciones, skew entre training y producción.
    - Detección de *drift* de datos o del concepto: variaciones en la distribución de entrada o en la relación entre inputs y outputs que degradan el modelo.
    - Feedback de negocio: impacto real de las predicciones en métricas de negocio.
    - Registro y rastreo de versiones: datos, modelo, configuración, entorno de ejecución.
    
    El módulo subraya que sin monitorización activa, los sistemas ML tienden a degradarse silenciosamente.
    
    ---
    
    ## 6. Preguntas que debes plantearte
    
    El módulo también propone una lista de preguntas estratégicas que el equipo de ML debe hacerse, por ejemplo:
    
    - ¿Cuándo deben actualizarse los datos y el modelo?
    - ¿Qué latencia es aceptable para la inferencia?
    - ¿Cómo vamos a versionar los modelos, los datos y las transformaciones?
    - ¿Cómo vamos a medir el impacto del modelo en el negocio y detectar fallos?
    - ¿Qué sucede si la distribución de datos cambia o el servicio sufre un fallo?
    
    Estas preguntas fomentan una mentalidad de ingeniería ML orientada al negocio, no solo a la precisión del modelo.
    
    ---
    
    ## 7. Conexión con tus prácticas académicas y profesionales (relevancia para tu portafolio)
    
    Dado tu contexto (Dirección de Empresas + Auditoría y Assurance en ML/IA), este módulo te aporta:
    
    - Una visión de **arquitectura y gobernanza de sistemas ML** que complementa tu comprensión técnica de modelos (YOLOv8, MLP, etc.).
    - Un puente entre tu formación en IA/ML y la dimensión operativa/jurídica: auditoría de modelos, control de calidad, monitorización continua.
    - Un marco para entender cómo los temas de **ética**, **sostenibilidad de modelos**, **riesgos operativos** y **marcos normativos (como IFRS S1, ESRS G1)** pueden integrarse en un sistema ML de producción.
    - Una base para diseñar portfolios de IA que no solo validen modelos, sino que también consideren su implementación, mantenimiento, y gobernanza.
    
    ---
    
    ## 8. Elementos clave para tu Master/Portafolio
    
    Cuando incorpores este módulo en tu portafolio, te sugiero destacar:
    
    - **Arquitectura completa de sistemas ML**: enfatizá que el modelo es solo una parte minoritaria del sistema.
    - **Decisiones de diseño**: estático vs dinámico en entrenamiento e inferencia; transformaciones de datos en tiempo de entrenamiento vs inferencia; testing y monitorización.
    - **Riesgos operativos**: drift de datos, skew, latencia, dependencia de datos, errores de producción.
    - **Buenas prácticas de MLOps**: versionado, reproducibilidad, monitorización, métricas de negocio, impacto organizacional.
    - **Enlace con auditoría / dirección de empresas**: cómo los sistemas ML deben estar integrados en controles, cumplimiento, sostenibilidad, gobernanza.
    
    ---
    
    ## 9. Síntesis final
    
    El módulo “Production ML Systems” del Google ML Crash Course proporciona un panorama exhaustivo sobre lo que realmente significa llevar un modelo de machine learning al entorno de producción y operarlo con éxito. Desde decisiones de arquitectura (entrenamiento/inferencia estática vs dinámica), pasando por **transformaciones de datos**, testing, despliegue, hasta la monitorización operativa, el módulo organiza los desafíos técnicos, de ingeniería y de negocio que se presentan en sistemas ML de vida real.
    
    Para ti, Juli, que estructuras proyectos de IA complejos (como fine-tuning de Transformers, portafolios de ML, auditoría de modelos), este módulo actúa como **puente** entre el desarrollo académico/técnico y la implementación empresarial/organizada. Incorporarlo en tu portafolio fortalece tu narrativa sobre no solo “hacer modelos” sino “hacer modelos que funcionen, se mantengan y generen valor”.
    
2. **MLOps: Continuous Delivery and Automation Pipelines in Machine Learning** – Google Cloud Architecture Center
    
    [https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
    
    ## Introducción
    
    El documento aborda cómo aplicar las prácticas de DevOps de forma adaptada al ciclo de vida de los sistemas de machine learning (ML), es decir, cómo construir pipelines de CI/CD/CT (Integración Continua, Entrega Continua, Entrenamiento Continuo) para modelos de ML en producción. [Google Cloud Documentation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
    
    Presenta un marco de referencia para la automatización, gestión de datos, verificación, deployment, monitoreo y gobernanza de modelos ML.
    
    ---
    
    ## 1. DevOps vs. MLOps
    
    Se diferencia claramente entre DevOps (orientado al código de software) y MLOps (orientado al modelo + datos + infraestructura).
    
    - DevOps aborda ciclos de desarrollo rápidos, pruebas automáticas, integración en producción. [Google Cloud Documentation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
    - MLOps incorpora además: la gestión de datos, el entrenamiento de modelos, el serving, el monitoreo de predicciones, la detección de *drift*.
        
        El documento recalca que **el ML pipeline** tiene componentes adicionales al software tradicional.
        
    
    ---
    
    ## 2. Pasos del desarrollo de modelos ML
    
    El artículo describe los bloques típicos de un pipeline ML:
    
    - Recolección de datos y verificación (data ingestion + data validation)
    - Preparación (feature engineering, normalización, transformación)
    - Entrenamiento del modelo
    - Evaluación y validación del modelo
    - Despliegue del modelo (serving)
    - Monitorización y mantenimiento del modelo en producción (latencia, precisión, drift)
    - Gestión de recursos, versión de datos y modelo, metadatos. [Google Cloud Documentation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
    
    Este flujo no es lineal, sino iterativo: los modelos pueden requerir reentrenamiento, nuevos datos o if–updates.
    
    ---
    
    ## 3. Niveles de madurez de MLOps
    
    El documento define varios *niveles de madurez* para pipelines de ML:
    
    - Nivel 0: procesos manuales, sin automatización real.
    - Nivel 1 y superiores: introducción de automatización, CI/CD, retraining automático. [Google Cloud Documentation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
        
        Cada nivel implica mayor capacidad de escalar, reproducir, desplegar de forma confiable.
        
        El objetivo es que la organización avance hacia pipelines más maduros para reducir errores, acelerar producción y mitigar riesgos.
        
    
    ---
    
    ## 4. Entrega continua y entrenamiento continuo
    
    ### 4.1 Entrega continua (Continuous Delivery)
    
    Se refiere a la capacidad de poner en producción cambios en código, configuración o modelos de forma segura, frecuente y reproducible. [Google Cloud Documentation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
    
    Incluye tests automatizados, pipelines de despliegue, versionamiento y rollback.
    
    ### 4.2 Entrenamiento continuo (Continuous Training)
    
    Es un aspecto más específico de MLOps: cuando los datos cambian o emergen nuevas condiciones, el modelo debe ser reentrenado automáticamente. Esto permite mitigar *data drift* y *concept drift*. [Google Cloud Documentation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
    
    La combinación CD + CT da lugar a sistemas ML verdaderamente adaptativos.
    
    ---
    
    ## 5. Automatización del pipeline
    
    El artículo enfatiza que la automatización es clave para escalar. Algunos puntos destacados:
    
    - Uso de contenedores, orquestadores (por ejemplo Kubernetes) y workflows para ML.
    - Automatización de pasos: ingestión de datos, entrenamiento, evaluación, test, deployment.
    - Validación de datos, test del modelo, pruebas de regresión, monitorización.
    - Versionamiento de código, datos, modelos, transformaciones.
        
        Sin automatización, los sistemas ML tienden a fallar por falta de reproducibilidad y despliegue manual. [Google Cloud Documentation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
        
    
    ---
    
    ## 6. Gestión de metadatos, versionado y reproducibilidad
    
    Para sostener un sistema ML en producción se requiere:
    
    - Versionar los datos de entrenamiento, los artefactos del modelo, la configuración del pipeline.
    - Rastrear los experimentos, los hyper-parámetros, métricas y entorno de ejecución.
    - Asegurar que el training-serving skew esté controlado (que el entorno de entrenamiento sea lo más cercano posible al de inferencia).
        
        El documento alerta que la deuda técnica en ML (data dependencies, transformaciones desalineadas, modelos sin monitoreo) es mayor que en software clásico. [Google Cloud Documentation+1](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
        
    
    ---
    
    ## 7. Monitorización y mantenimiento del modelo en producción
    
    El mantenimiento no termina al desplegar el modelo. El documento señala que es esencial:
    
    - Monitorear latencia de inferencia, tasa de errores, distribución de predicciones, anomalías en los datos de entrada.
    - Detectar *data drift* y *concept drift*.
    - Disponer de mecanismos de alerta y retraining (o rollback).
    - Medir el impacto de negocio de las predicciones, no solo métricas técnicas.
        
        Sin una monitorización robusta, los modelos se degradan silenciosamente. [Google Cloud Documentation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
        
    
    ---
    
    ## 8. Consideraciones especiales para sistemas generativos / IA grande escala
    
    El documento menciona que para modelos generativos o de gran escala (LLMs, multimodal) los retos de MLOps se incrementan: latencia, costo de inferencia, requisitos de hardware, ética, explicabilidad, reutilización de embeddings, pipelines de datos multimodales. [Google Cloud Documentation](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
    
    ---
    
    ## 9. Relevancia para tu perfil profesional
    
    Dado tu interés en IA/ML, auditoría, ética y frameworks como IFRS/ESRS, este artículo aporta:
    
    - Un marco lógico para **integrar modelos IA en sistemas reales**, lo cual es clave para asegurar gobernanza, transparencia y ética.
    - Una visión de cómo **MLOps opera como puente entre ciencia de datos y operaciones empresariales**, que complementa tu rol en auditoría & assurance.
    - Una base para interpretar cómo **los riesgos operativos de IA (drift, sesgo, reproducibilidad) pueden controlarse** mediante pipelines automatizados.
    - Una guía para diseñar y documentar tus proyectos de IA no solo en fase experimental, sino también para producción, mantenimiento y reporte.
    
    ---
    
    ## 10. Elementos destacados para tu portafolio
    
    Te recomiendo resaltar:
    
    - La arquitectura de pipeline (ingestión → entrenamiento → despliegue → monitorización).
    - Los artefactos clave: análisis de datos, transformación, modelo, versión, métricas, reporte.
    - Las fases de madurez de MLOps y cómo tu trabajo puede moverse hacia niveles más altos (automatización, retraining, monitorización continua).
    - Las implicaciones de auditoría/ética: reproducibilidad, control de sesgo, monitoreo de impacto, gobierno de modelos.
3. **Rules of Machine Learning: Best Practices for ML Engineering** – Google Developers
    
    [https://developers.google.com/machine-learning/guides/rules-of-ml](https://developers.google.com/machine-learning/guides/rules-of-ml)
    
    # **1. Visión General del Documento**
    
    Las *Rules of ML* no enseñan a entrenar modelos, sino a **evitar los errores más comunes de ingeniería**, priorizando:
    
    - Simplicidad del pipeline
    - Iteración rápida
    - Mantenibilidad
    - Priorización de señales robustas sobre complejidad del modelo
    - Evaluación y métricas confiables
    - Gestión del riesgo y del ciclo de vida del sistema
    
    Estas reglas pretenden evitar que los sistemas ML “colapsen bajo su propio peso” debido a mala arquitectura, mala gestión de datos o modelos demasiado acoplados.
    
    ---
    
    # **2. Primero, haz que funcione → luego mejora el modelo**
    
    Google enfatiza que **el mayor error de los equipos es complicar el sistema demasiado temprano**.
    
    Las reglas iniciales recomiendan:
    
    ### **Regla 1 — No empieces con un modelo complejo**
    
    Comenzar con un baseline simple (regresión logística, árboles, heurísticas) permite:
    
    - Iterar más rápido
    - Detectar errores de datos
    - Capturar señales básicas
    - Establecer métricas iniciales reales
    
    ### **Regla 2 — Trace lo más simple posible**
    
    Un pipeline simple permite detectar errores antes de escalar.
    
    Evita acoplar múltiples transformaciones desde el inicio.
    
    ---
    
    # **3. Gestión de Datos: el corazón del sistema**
    
    Google afirma que **los datos importan más que los modelos**.
    
    Varias reglas se enfocan en sanidad de datos:
    
    ### **Regla 3 — Verifica rigurosamente los datos**
    
    - Validación de rangos
    - Chequeos de NaNs
    - Detección de outliers
    - Consistencia entre entrenamiento e inferencia
        
        Errores en los datos suelen explicar la mayoría de las degradaciones.
        
    
    ### **Regla 4 — Mantén features y labels trazables**
    
    No mezclar fuentes, registrar origen y mantener versionado.
    
    ### **Regla 5 — Evita features que dependan del futuro**
    
    Previene fuga de información (*data leakage*), uno de los errores más frecuentes.
    
    ---
    
    # **4. Evitar el “Training-Serving Skew”**
    
    Una de las advertencias más repetidas:
    
    ### **Regla 6 — Asegura que las transformaciones sean idénticas en training e inferencia**
    
    Diferencias mínimas entre ambos entornos producen degradaciones invisibles:
    
    - Distintas librerías
    - Normalizaciones inconsistentes
    - Features faltantes
    - Preprocesamientos que no están en producción
    
    Recomendación: empaquetar features, transformaciones y modelos en artefactos reproducibles.
    
    ---
    
    # **5. Construcción de Features y Señales**
    
    ### **Regla 7 — Agregá features solo cuando sea necesario**
    
    Google advierte del “feature bloat”: demasiadas features hacen el modelo inestable y difícil de mantener.
    
    ### **Regla 8 — Analiza la correlación y la importancia**
    
    La ingeniería de features debe guiarse por evidencia, no por intuición.
    
    ### **Regla 9 — Prefiere features simples y robustas**
    
    Features complejas (modelos dentro de modelos) generan dependencias frágiles y riesgo técnico.
    
    ---
    
    # **6. Métricas y evaluación**
    
    ### **Regla 10 — Elegí una métrica principal y mantenela estable**
    
    No mezclar múltiples métricas sin priorizar cuál representa el objetivo real del negocio.
    
    ### **Regla 11 — Medí continuamente el desempeño en producción**
    
    El modelo puede tener buen performance offline pero fallar por drift, cambios en usuarios o cambios en distribución.
    
    ### **Regla 12 — Tené un conjunto de validación que represente producción**
    
    Los equipos suelen usar datasets desfasados o artificiales. Google insiste en usar datos frescos y reales.
    
    ---
    
    # **7. Evitar sobreajuste organizacional (no solo técnico)**
    
    Uno de los aportes más interesantes:
    
    ### **Regla 13 — No optimices tu modelo para un dataset que pronto dejará de existir**
    
    Sucede frecuentemente cuando:
    
    - Se optimiza en datasets demasiado pequeños
    - Se entrena en condiciones ideales que no existen en producción
    - El dataset es muy estático y se vuelve obsoleto
    
    ---
    
    # **8. Automatización y MLOps**
    
    ### **Regla 14 — Automatiza el pipeline lo antes posible**
    
    Incluye:
    
    - Validación de datos
    - Reentrenamiento
    - Evaluación de modelo
    - Pruebas
    - Despliegue
    
    ### **Regla 15 — Monitorea datos y predicciones en producción**
    
    El modelo debe ser observado de forma continua:
    
    - Drift
    - Latencia
    - Errores
    - Cambios súbitos en distribución
    - Estabilidad de features
    
    ---
    
    # **9. Gestión de Riesgo y Deuda Técnica**
    
    Google es explícito:
    
    **Los sistemas ML acumulan deuda técnica más rápido que el software tradicional.**
    
    ### Riesgos típicos:
    
    - Features dependientes entre sí
    - Procesos manuales
    - Falta de versionado
    - Datos no gobernados
    - Falta de monitoreo
    - Falta de reproducibilidad
    
    ### **Regla 16 — Minimiza dependencias externas**
    
    Si una feature depende de otra API externa no controlada, aumenta el riesgo sistémico.
    
    ---
    
    # **10. Escalabilidad y evolución del sistema**
    
    ### **Regla 17 — No agregues complejidad antes de tiempo**
    
    Evitar redes neuronales enormes cuando un modelo tradicional es suficiente.
    
    ### **Regla 18 — Evalúa el costo-beneficio de cada feature adicional**
    
    Cada feature nueva:
    
    - Reduce interpretabilidad
    - Aumenta riesgo
    - Agrega mantenimiento
    - Aumenta latencia
    
    ---
    
    # **11. Cultura: ML como ingeniería, no solo experimentación**
    
    Google concluye que los equipos deben:
    
    - Documentar decisiones
    - Trabajar con reproducibilidad
    - Tener procesos de revisión
    - Entender ML como software productivo
    - Alinear métricas técnicas con objetivos de negocio y ética
    
    ---
    
    # **12. Conexión con tu portafolio y perfil profesional**
    
    Este documento complementa tu práctica en:
    
    ### **Dirección de Empresas + IA/ML**
    
    - Relaciona diseño de modelos con impacto en negocio.
    - Proporciona criterios para asegurar calidad, gobernanza y explicabilidad (clave en auditoría).
    
    ### **Trabajo en Deloitte & Auditoría**
    
    - Las reglas se alinean con los principios de control interno, monitoreo continuo, versionado y reproducibilidad.
    - Podés vincularlas con normas como ESRS, IFRS S1 y criterios de governanza de modelos.
    
    ### **Proyectos personales en IA**
    
    Te sirven para justificar decisiones en tus portafolios:
    
    - Por qué empezaste con modelos sencillos
    - Cómo controlás el skew
    - Por qué priorizás features robustas
    - Cómo diseñaste pipelines reproducibles
    
    ---
    
    # **Conclusión General**
    
    “Rules of Machine Learning” es una referencia esencial para cualquier profesional que trabaje con ML en producción. Más que presentar algoritmos, enseña **cómo evitar errores graves** y diseñar sistemas escalables, gobernables y robustos.
    
    Es un texto que vincula la ingeniería de software, MLOps y la gestión del ciclo de vida de modelos con la visión de negocio, lo que encaja perfectamente con tu perfil académico y profesional.
    
4. **Hidden Technical Debt in Machine Learning Systems** – Sculley et al., NIPS 2015
    
    [https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)
    

# **1. Motivación principal del paper**

Los autores sostienen que:

- El machine learning introduce **formas únicas de deuda técnica** que no aparecen en software convencional.
- Estos problemas no suelen estar visibles al comienzo, pero crecen rápidamente y pueden bloquear actualizaciones, causar fallos graves o impedir auditorías y gobernanza.
- A medida que aumenta la escala del sistema, aumenta la fragilidad y el riesgo sistémico.

El mensaje central es: **la complejidad del sistema ML crece más rápido que la complejidad del modelo.**

---

# **2. Tipos de deuda técnica en sistemas de ML**

El artículo categoriza varias fuentes de deuda técnica “oculta” que afectan la mantenibilidad, la reproducibilidad y la seguridad del sistema.

---

## **2.1. Entrelazamiento (Entanglement)**

Característica única del ML:

**Pequeños cambios en una parte del sistema pueden alterar el comportamiento global del modelo.**

Ejemplos:

- una feature afecta a muchas otras,
- cambiar una columna altera el comportamiento no lineal del modelo,
- ajustes locales generan efectos colaterales globales.

Esto hace muy difícil localizar la causa de una degradación.

---

## **2.2. Dependencias complejas (Complex Dependencies)**

Los modelos dependen de:

- múltiples fuentes de datos,
- versiones de datasets,
- pipelines de transformación,
- servicios externos,
- features que dependen entre sí.

Estas dependencias pueden “pudrirse”, romperse o cambiar sin aviso (data drift, schema drift), generando fallos difíciles de diagnosticar.

---

## **2.3. Deuda de features (Data & Feature Debt)**

Sculley describe que **cada feature añadida genera deuda técnica**, incluso si mejora performance.

Problemas típicos:

- features duplicadas,
- features muertas (“zombie”),
- features creadas para un dataset que ya no existe,
- features que dependen del futuro (data leakage),
- “cascadas” de features que dependen una de otra.

Cada feature nueva aumenta el costo de mantenimiento del sistema entero.

---

## **2.4. Deuda por configuración del sistema (Configuration Debt)**

Los sistemas ML suelen tener:

- configuraciones complejas,
- parámetros de preprocesamiento,
- pipelines ajustados manualmente,
- scripts específicos para cada paso.

Sin un sistema de configuración centralizado, la deuda explota.

---

## **2.5. Deuda de implementación por glue code**

“Glue code” = código que conecta componentes incompatibles.

Es uno de los principales generadores de deuda técnica en ML, porque:

- se produce en grandes cantidades,
- suele ser frágil,
- depende de APIs cambiantes,
- multiplica puntos de fallo.

Google describe pipelines ML con cientos de archivos dedicados solo a glue code.

---

## **2.6. Deuda por falta de aislamiento: *Pipeline Jungles***

Se describe cómo los pipelines de datos crecen orgánicamente:

- múltiples jobs dispersos,
- ETLs sin versionar,
- scripts manuales,
- duplicación de transformaciones.

Esto produce sistemas opacos, difíciles de reproducir y prácticamente imposibles de auditar.

---

## **2.7. Deuda por feedback loops no controlados**

El ML puede influir sobre los datos futuros (ej. recomendaciones, ads, decisiones crediticias).

Si no se controla:

- el modelo **se autoalimentará**,
- amplificará sesgos,
- degradará los datos,
- y sesgará la distribución original.

Este es uno de los riesgos más graves, especialmente en sistemas de high-stakes (finanzas, salud, crédito).

---

# **3. Monitoreo insuficiente y falta de métricas**

Los autores insisten en que:

- Sin monitoreo continuo, el modelo se degrada silenciosamente.
- Cambios en la distribución de entrada (data drift) o en la relación input-output (concept drift) son inevitables.
- La deuda aparece cuando las métricas no capturan el comportamiento real en producción.

Google advierte que la falta de observabilidad es uno de los errores más costosos.

---

# **4. Costos organizacionales y deuda social**

La deuda técnica en ML no es solo técnica:

- equipos que dependen de un único “experto en features”,
- dificultades para entrenar a nuevos miembros,
- falta de documentación,
- pipeline imposible de reproducir o auditar.

La complejidad del sistema genera una “deuda organizacional” que afecta productividad y confiabilidad.

---

# **5. Recomendaciones de mitigación**

El paper propone principios clave para reducir deuda:

### ✔️ Simplificar features

Eliminar features obsoletas, redundantes o costosas.

### ✔️ Centralizar y versionar el pipeline

Transformaciones, datos, modelos, configuraciones, todo debe ser trazable.

### ✔️ Aislar componentes

Evitar que las transformaciones afecten múltiples partes del sistema de forma oculta.

### ✔️ Monitorear activamente

Detectar drift, degradación y anomalías.

### ✔️ Priorizar reproducibilidad

Pipelines declarativos, artefactos versionados, ambientes replicables.

### ✔️ Adoptar MLOps

Automatización, pruebas, CI/CD, evaluación continua.

---

# **6. Relevancia para tu portafolio, auditoría y ética**

Este paper es directamente aplicable a tus proyectos y a tu perfil en Deloitte:

### **Para auditoría y assurance en IA**

- Explica dónde ocurren los riesgos técnicos de los modelos.
- Ofrece guías para evaluar gobernanza, reproducibilidad y trazabilidad.
- Conecta con principios de ESRS, IFRS S1/S2, G1 (riesgo operacional, sostenibilidad de modelos).

### **Para tus portafolios de IA (YOLO, Transformers, RAG)**

- Te da vocabulario para justificar decisiones de diseño.
- Te permite explicar por qué la complejidad del sistema importa más que la del modelo.
- Te posiciona como alguien que entiende IA más allá del modelo (visión de sistemas).

### **Para tu futuro profesional**

Este paper es uno de los textos más citados en MLOps y gobernanza de IA.

Dominarlo es clave para roles en IA responsable, data governance, auditoría algorítmica y desarrollo de sistemas ML robustos.

---

# **Conclusión General**

*Hidden Technical Debt in ML Systems* es un paper fundamental para entender por qué los modelos no fallan aislados, sino como parte de sistemas complejos, frágiles y altamente interconectados.

La deuda técnica en ML es más peligrosa porque está oculta y crece más rápido que en software tradicional.

La solución no es tener mejor accuracy, sino tener **mejor ingeniería, mejor gobernanza y mejores sistemas**.