# Auditoría metodológica del Laboratorio ETL

**Proyecto:** Titanic Survival Analytics — Cristian Camilo Guevara (UTN)

## Integridad verificada

- Datos originales: 891 registros, 12 columnas.
- Datos procesados: 891 registros, 21 columnas.
- `PassengerId` y `Survived` se mantienen idénticos, fila por fila.
- `Age`: 177 faltantes originales, 0 procesados.
- `Embarked`: 2 faltantes originales, 0 procesados.
- `Cabin`: 687 faltantes originales, 687 procesados.

## Recorrido pedagógico implementado

1. **Radiografía:** gráfico Antes/Después de las tres columnas con valores faltantes, etiquetas y explicación del motivo de cada decisión.
2. **Mapa de ausencia:** mapa de calor interactivo por pasajero, variable y versión; los tooltips indican presente/faltante.
3. **Microscopio:** consulta de registros individuales antes y después; distingue imputaciones de valores observados.
4. **Simulador:** compara mediana por grupos (versión oficial), mediana global, media global y eliminación de filas. Los experimentos no alteran los CSV.
5. **Feature engineering:** tabla de atributos derivados y gráficos interactivos de `FamilySize` y `Title`.

## Interpretación y limitaciones

- Imputar edad no recupera un valor histórico real: produce una estimación estadística.
- La ausencia masiva de cabina impide reconstruir cabinas individuales; mantener `Cabin` vacío evita fabricar evidencia.
- El tratamiento de datos puede afectar la distribución de edad y, al eliminar filas, el tamaño y composición de la muestra.
- Los filtros globales afectan Historia y EDA; el laboratorio ETL trabaja con el conjunto completo para mantener comparabilidad.
- El simulador de ML es educativo y su pipeline de imputación se ajusta solo con datos de entrenamiento.

## Pruebas realizadas

- `python validar_proyecto.py`: **OK**.
- Inspección del código de la pestaña ETL y de las rutas de los datos: **OK**.
- **Pendiente:** recorrido visual completo de los controles en el navegador y despliegue remoto en Streamlit Cloud.

## Mejoras recomendadas para la exposición

- Explicar que las 9 columnas derivadas no equivalen a 9 nuevos pasajeros.
- Distinguir valores observados e imputados en todo gráfico de edades.
- Indicar que los diagramas de supervivencia muestran asociaciones y no causalidad.