# Reporte del Modelo Baseline

Este documento contiene los resultados del modelo baseline.

## Descripción del modelo

Se utiliza una CNN para la extracción de características visuales de imágenes y una LSTM para la generación de descripciones textuales.

## Variables de entrada

Imágenes y descripciones

## Variable objetivo

Nueva descripción de la imagen indicada.

## Evaluación del modelo

### Métricas de evaluación

BLEU Score: Mide la precisión de las secuencias de palabras generadas.

METEOR Score: Evalúa la calidad de la traducción considerando sinónimos y variaciones gramaticales.

### Resultados de evaluación

BLEU score para el primer modelo: 0.022740541385519286

BLEU score para el segundo modelo: 0.009111110063993637


METEOR score para el primer modelo: 0.15366109893445237

METEOR score para el segundo modelo: 0.12801399010458697

## Análisis de los resultados

El modelo baseline ha mostrado que se puede utilizar para generar descripciones, aunque con margen de mejora (las métricas aunque son mejores que el segundo modelo, aún dan valores bajos).

## Conclusiones

El modelo baseline (modelo 1) genera de forma eficaz descripciones básicas de la imagen, pero requiere mejoras.

## Referencias

https://www.kaggle.com/datasets/adityajn105/flickr8k
