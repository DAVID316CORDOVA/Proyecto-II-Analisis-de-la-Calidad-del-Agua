# Calidad del Agua en India — Análisis con PySpark, Keras y MLlib

**Pontificia Universidad Javeriana · Procesamiento de Alto Volumen de Datos**  
**Autor:** Félix Córdova  
**Notebook:** `Clean_ML_Water_Cordova_Felix.ipynb`

---

## Descripción del Proyecto

Este proyecto aplica técnicas de **Procesamiento de Datos en Alto Volumen (PAVD)** con PySpark
para diagnosticar y predecir la calidad del agua en los ríos de la India. Se implementan y
comparan dos modelos de machine learning:

- **Modelo 1:** Red Neuronal Densa (Keras / TensorFlow)
- **Modelo 2:** Random Forest Regressor (MLlib nativo de Apache Spark)

La variable objetivo es el **Índice de Calidad del Agua (WQI)**, calculado a partir de seis
parámetros fisicoquímicos y bacteriológicos según la metodología de la referencia bibliográfica.

**Referencia de parámetros:** https://www.intechopen.com/chapters/69568

---

## Estructura del Repositorio

```
tarea_mpi/
│
├── Clean_ML_Water_Cordova_Felix.ipynb   ← Notebook principal
├── waterquality.csv                     ← Dataset 
├── Indian_States.shp / .dbf / .shx ...  ← Shapefile del mapa de India
│
├── *.PNG                                ← Imágenes exportadas del notebook
│
└── README.md
```

---

## Metodología

```
1. Carga de datos desde HADOOP HDFS
       ↓
2. Análisis exploratorio (EDA) + limpieza de nulos
       ↓
3. Conversión de tipos y eliminación de columnas irrelevantes
       ↓
4. Cálculo de rangos de calidad por parámetro (qrPH, qrDO, qrCOND, qrBOD, qrNN, qrFecal)
       ↓
5. Cálculo del WQI ponderado + clasificación (Excelente / Buena / Baja / Muy_Baja / Inadecuada)
       ↓
6. Visualización geoespacial (mapa coroplético de India)
       ↓
7. Entrenamiento Modelo 1: Red Neuronal Keras
       ↓
8. Entrenamiento Modelo 2: Random Forest MLlib
       ↓
9. Comparación de modelos (MSE, RMSE, MAE, R²)
```

---

## Dataset

| Campo | Descripción |
|-------|-------------|
| `STATION CODE` | Código de estación de medición |
| `LOCATIONS` | Ubicación del río |
| `STATE` | Estado de la India |
| `TEMP` | Temperatura del agua (°C) |
| `DO` | Oxígeno Disuelto (mg/L) |
| `pH` | Acidez del agua |
| `CONDUCTIVITY` | Conductividad eléctrica (µS/cm) |
| `BOD` | Demanda Bioquímica de Oxígeno (mg/L) |
| `NITRATE_N_NITRITE_N` | Nitratos/Nitritos (mg/L) |
| `FECAL_COLIFORM` | Coliformes Fecales (UFC/100mL) |
| `TOTAL_COLIFORM` | Eliminada (no aporta al modelo) |

**Total de registros:** 534 · **División:** 80% entrenamiento / 20% prueba

---

## 1. Análisis Exploratorio (EDA)

### Parámetros DO y pH

![](<parametro do y ph de la caldiad del agua.PNG>)

> Los parámetros Oxígeno Disuelto y pH muestran alta variabilidad entre estaciones, con valores
> de DO que en varias zonas caen por debajo de 6 mg/L, indicando deterioro de la calidad del agua.

---

### Parámetros BOD y Nitratos/Nitritos

![](<parametro bod y nitrogeno.PNG>)

> La Demanda Bioquímica de Oxígeno presenta picos extremos en ciertas estaciones, asociados a
> zonas con alta descarga de materia orgánica. Los Nitratos muestran concentraciones generalmente
> bajas (< 20 mg/L) en la mayoría de los estados.

---

### Parámetros Conductividad y Coliformes Fecales

![](<parametro conductividad.PNG>)

> Los Coliformes Fecales exhiben la mayor dispersión de todos los parámetros — valores extremos
> en estados del norte de la India. Esto anticipa su alta importancia relativa en el modelo RF.

---

## 2. Mapa Geoespacial de la India

### Mapa Base

![](<mapa inicial de la india.PNG>)

---

### Mapa Coroplético — Índice WQI por Estado

![](<indice de calidad wqi.PNG>)

> Los estados del noreste presentan mejores índices WQI. Los estados del cinturón industrial
> del norte (Uttar Pradesh, Bihar) muestran los valores más críticos. Los estados en gris
> claro no tienen datos disponibles en el dataset.

---

### Histograma WQI por Estado

![](<histograma de calidad.PNG>)

> La mayoría de las estaciones de medición caen en la categoría **"Baja"** (WQI entre 50 y 75),
> lo que implica que el agua requiere tratamiento antes del consumo humano.

---

## 3. Modelo 1 — Red Neuronal Keras

### Arquitectura

![](<model sequential.PNG>)

```
Capa de entrada:  Dense(350, input_dim=6, activation='relu')
Capa oculta 1:    Dense(350, activation='relu')
Capa oculta 2:    Dense(350, activation='relu')
Capa de salida:   Dense(1,   activation='linear')

Total parámetros entrenables: 248,501 (970.71 KB)
Optimizador: Adam (lr=0.001) | Función de pérdida: MSE
Épocas: 200 | Batch size: 81
```

### Curva de Pérdida

![](<curva de perdida red neuronal.PNG>)

> La curva de pérdida converge progresivamente, sin oscilaciones significativas, lo que
> indica un entrenamiento estable y un aprendizaje efectivo de la relación entre los
> rangos de calidad y el WQI.

### Valores Reales vs. Predichos (Keras)

![](<red neuronal valores reales vs predichos.PNG>)

> Los puntos se distribuyen muy cercanos a la línea ideal (diagonal roja), confirmando
> el excelente ajuste del modelo sobre el conjunto de prueba.

---

## 4. Modelo 2 — Random Forest MLlib (PySpark)

```python
RandomForestRegressor(
    featuresCol = 'features',
    labelCol    = 'WQI',
    numTrees    = 100,
    maxDepth    = 10,
    seed        = 1
)
```

> **Ventaja clave:** Opera de forma nativa y distribuida sobre Spark sin necesitar
> convertir los datos a pandas, lo que lo hace más escalable para Big Data real.

### Importancia de Variables

![](<importancia de las variables.PNG>)

| Variable | Importancia |
|----------|-------------|
| `qrFecal` | **43.56%** |
| `qrDO`    | **28.64%** |
| `qrCOND`  | 14.60% |
| `qrBOD`   | 8.30% |
| `qrPH`    | 4.67% |
| `qrNN`    | 0.23% |

> Los Coliformes Fecales y el Oxígeno Disuelto son los parámetros más determinantes
> del WQI, coherente con sus pesos en la fórmula original (0.281 cada uno).

### Valores Reales vs. Predichos (Random Forest)

![](<random forest valroes reales vs predichos.PNG>)

> Buen ajuste general, aunque con mayor dispersión alrededor de la línea ideal
> respecto a la Red Neuronal, especialmente en valores extremos de WQI.

---

## 5. Comparación de Modelos

![](<comparacion de errores.PNG>)

| Métrica | Red Neuronal Keras | Random Forest MLlib |
|---------|:-----------------:|:-------------------:|
| MSE     | **0.2614**        | 9.1848              |
| RMSE    | **0.5112**        | 3.0306              |
| MAE     | **0.1362**        | 2.0920              |
| R²      | **0.9990**        | 0.9586              |

**Ganador por métricas:** Red Neuronal Keras (menor error en todas las métricas, R² = 99.90%)

### ¿Por qué Keras supera a Random Forest aquí?

El WQI es una combinación lineal ponderada de los rangos de calidad. Las redes neuronales
con descenso de gradiente pueden aproximar esa estructura con altísima precisión. Random Forest,
al operar con particiones discretas del espacio de características, introduce mayor error
cuantitativo, aunque sigue siendo un modelo muy competitivo (R² = 95.86%).

### ¿Cuándo usar Random Forest sobre Keras?

En entornos de producción Big Data con millones de registros distribuidos en un clúster,
el hecho de que MLlib no requiera `toPandas()` representa una ventaja arquitectónica
que justifica su uso, aun con la diferencia de precisión observada.

---

## Resultados Finales

```
Red Neuronal Keras:
  MSE  = 0.2614  |  RMSE = 0.5112  |  MAE = 0.1362  |  R² = 0.9990

Random Forest MLlib:
  MSE  = 9.1848  |  RMSE = 3.0306  |  MAE = 2.0920  |  R² = 0.9586
```

---

## Requisitos del Entorno

```
Apache Spark   >= 3.5
PySpark        >= 3.5
Python         >= 3.12
TensorFlow / Keras
scikit-learn
geopandas
pandas
numpy
matplotlib
seaborn
findspark
adjustText
mapclassify    >= 2.4.0
```

> El notebook fue ejecutado sobre el clúster Hadoop/Spark de la Pontificia Universidad Javeriana.
> La ruta HDFS del dataset: `hdfs://10.195.34.34:9000/csv/waterquality.csv`

---

## Referencias

1. Brown, R. M., McClelland, N. I., Deininger, R. A., & Tozer, R. G. (1970). A water quality index: do we dare? *Water & Sewage Works*, 117(10), 339–343.

2. Tyagi, S. et al. (2013). Water quality assessment in terms of water quality index. En *Water Quality — Science, Assessments and Policy*. IntechOpen. https://www.intechopen.com/chapters/69568

3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

4. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*. https://arxiv.org/abs/1412.6980

5. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444. https://doi.org/10.1038/nature14539

6. Zaharia, M. et al. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM*, 59(11), 56–65. https://doi.org/10.1145/2934664

7. Apache Spark (2024). MLlib — Machine Learning Library. https://spark.apache.org/docs/latest/ml-classification-regression.html

8. Chollet, F. et al. (2015). Keras. https://keras.io

9. GeoPandas Development Team (2024). GeoPandas Documentation. https://geopandas.org/en/stable/
