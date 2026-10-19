# Titanic análisis de sobrevivientes  

**Autor:** Cristian Camilo Guevara · Diplomatura en Ciencia de Datos y Análisis Avanzado (UTN), 2026.

Dashboard Streamlit, con narrativa y análisis técnico, construido a partir de los CSV reales generados en las etapas 1–3 del proyecto educativo. Incluye navegación por cinco pestañas: historia, ETL, EDA, simulador ML y calidad/descargas.

## Ejecutar en VS Code / Windows

```powershell
cd Titanic_Etapa4
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit mostrará una dirección local, habitualmente `http://localhost:8501`.

## Qué contiene

- `app.py`: aplicación ejecutable, gráficos Plotly y modelo de regresión logística.
- `data/titanic_original.csv`: archivo fuente UTN, 891 × 12.
- `data/titanic_procesado.csv`: salida de la etapa 2, 891 × 21.
- `.streamlit/config.toml`: tema oscuro.
- `requirements.txt`: dependencias.
- `validar_proyecto.py`: pruebas de integridad de los datos y estructura.

## Navegación

1. **La historia:** supervivencia por sexo y clase, Sankey de puerto → clase → resultado.
2. **Laboratorio ETL:** nulos antes/después, experimento de imputación de edad, features derivadas.
3. **Exploración EDA:** violín de tarifas, boxplot de edades, pirámide de supervivencia, correlaciones y filtros cruzados.
4. **Simulador ML:** regresión logística con pipeline y train/test estratificado; gauge con probabilidad estimada y métricas.
5. **Calidad y descargas:** matriz de ausencia, trazabilidad y exportaciones CSV.

Los filtros laterales recalculan las métricas y visualizaciones de las pestañas **Historia y EDA**, y el CSV de selección. El laboratorio ETL muestra experimentos independientes sobre la base original; el simulador usa un modelo entrenado sobre la base original y no cambia con los filtros globales.

## Metodología de ML

La separación entrenamiento/prueba (75/25, estratificada, semilla 42) se realiza **antes** de imputar. La imputación de medianas/modas, estandarización y codificación one-hot ocurren dentro del pipeline entrenado exclusivamente con entrenamiento, para reducir fuga de información. Se informa accuracy y ROC-AUC del conjunto de prueba. No se afirma precisión histórica exacta: es un **simulador educativo** con sesgos y límites de representatividad.

## Procedencia y ética

Los datos son los del material UTN facilitado por el estudiante. La versión procesada preserva 891 registros, 21 variables, y conserva `Cabin` como faltante en lugar de inventar cabinas. No se usan imágenes ni datos personales adicionales de terceros. Las asociaciones observacionales no demuestran causalidad.

## Publicación

Subir el contenido de esta carpeta a un repositorio de GitHub y, en Streamlit Community Cloud, seleccionar `app.py` como archivo principal. Revisar previamente políticas del servicio y el contenido de los datos antes de hacer público el repositorio.

## Actualización ETL visual (versión 2)

La pestaña **Laboratorio ETL** incluye ahora:

- Comparación Antes/Después enfocada en Edad, Cabina y Embarque, con etiquetas e interpretación.
- Mapa de calor Plotly interactivo por pasajero y variable, con cambio entre datos crudos y procesados.
- Microscopio de pasajeros: comparación de registros originales y procesados y señalización de imputaciones.
- Simulador de imputación (mediana por grupos, mediana global, media global y eliminación de filas), sin alterar los CSV.
- Gráficos Plotly interactivos de distribución de edades, tamaños familiares y títulos.
- Descarga de tabla comparativa de valores faltantes.

### Cómo actualizar en tu PC sin reinstalar paquetes

1. Detené Streamlit en la terminal con **Ctrl+C**.
2. Extraé este ZIP en una carpeta nueva o reemplazá **solo `app.py`** en tu carpeta actual `Titanic_Etapa4`, con VS Code cerrado o sin cambios sin guardar en el archivo.
3. Ejecutá ` .\\.venv\\Scripts\\python.exe -m streamlit run app.py ` en la carpeta de proyecto donde ya instalaste las dependencias.

No hace falta reinstalar las librerías ni reemplazar `.venv`. Los gráficos son interactivos de Plotly, no fotografías históricas.
