# Cambios al sistema base (rama `kenya-mejoras-base`)

Correcciones al sistema base para que los números de los equipos sigan siendo
**comparables** entre sí y contra la base. Están en la rama `kenya-mejoras-base`; no se
suben a `main` hasta que Andrea los revise.

## 1. La muestra "fija" de test cambiaba al agregar películas

**Problema.** `muestra_test` (en `evaluar_resena.py`, la usan también la reseña y el cartel)
mezclaba los ids de test con un shuffle de pandas. Ese orden depende de *qué películas hay*:
si se reconstruye el dataset con películas nuevas de los equipos (la guía lo pide), la muestra
cambia y las métricas base (BERTScore 0.104, CLIPScore 27.8…) ya no se pueden comparar. Eso
rompe la regla 1 de la guía.

**Solución.** La muestra solo toma películas de test del dataset **base** (`fuente == "base"`,
que no cambia cuando los equipos agregan) y las ordena por un hash del id
(`md5(f"{SEMILLA}-{id}")`). Agregar películas de equipos ya no altera la muestra. Se probó con
datos simulados: la muestra de 200 es idéntica con películas nuevas y con otro orden.

**Consecuencia.** La muestra es distinta a la de las métricas base ya publicadas.
**Hay que volver a correr una vez la base** para tener las referencias nuevas:

```bash
python -m p2m.evaluar_resena --nombre base
python -m p2m.evaluar_cartel --nombre base
```

Después de eso, las referencias de la tabla del README (reseña y cartel) se actualizan con
estos números. Las del clasificador **no cambian** (usa todo el split de test, no esta muestra).
Ojo: si se agregan películas de equipos, el test del clasificador sí crece; para comparar,
siempre contra el mismo dataset.

## 2. Los evaluadores solo aceptaban una capa lineal

**Problema.** `evaluar_clasificador.py` y la clase `Clasificador` creaban `torch.nn.Linear`
fijo y también usaban siempre CLIP B/32. Un equipo que entrene un MLP, o cambie de CLIP,
rompía el evaluador y la demo.

**Solución.** El `.pt` ahora guarda la `arquitectura`, el `modelo_clip` y el nombre del archivo
de `embeddings` con que se entrenó. Se reconstruye con `construir_cabeza` / `cargar_cabeza`
(`p2m/clasificador.py`):

```python
{"tipo": "lineal"}                       # la base
{"tipo": "mlp", "ocultas": [512, 256]}   # MLP con ReLU y dropout (opcional "dropout")
```

- Los `.pt` anteriores (sin esos campos) siguen funcionando: se tratan como lineales con CLIP B/32.
- Para un equipo que quiera otra forma, agrega su tipo en `construir_cabeza`; el evaluador y
  la demo la reconstruyen sin más cambios.
- `Clasificador(...)` usa el CLIP guardado en el `.pt`; `evaluar_clasificador` usa los vectores
  guardados en el `.pt` (`cargar_split(split, embeddings)`).

**Cómo usarlo (equipo del clasificador).** Al entrenar una variante, guardar en el `.pt`:
`{"modelo_clip": ..., "embeddings": "<archivo.npz>", "arquitectura": {...}, "generos": ..., "capa": ..., "umbrales": ...}`.

## 3. El cartel se medía con el clasificador de otro equipo

**Problema.** `evaluar_cartel` usaba `Clasificador()`, que busca primero `clasificador_base.pt`
en la carpeta de salidas del equipo. Si el equipo del clasificador guardaba el suyo (o cambiaba
de CLIP), el F1 de género y el CLIPScore del equipo del cartel cambiaban **sin que el cartel
hubiera cambiado**.

**Solución.** `evaluar_cartel` mide siempre con el clasificador base de la carpeta del dataset
(`ruta_modelo_base()`), el "juez" fijo. Se puede cambiar a propósito con `--juez ruta.pt`.

## Qué se verificó y qué no

Verificado (sin el dataset, que vive en Drive):
- Todos los archivos de `p2m/` compilan.
- `construir_cabeza` / `cargar_cabeza`: un `.pt` antiguo se carga como lineal con la misma
  salida; un MLP se guarda y se reconstruye idéntico; un tipo desconocido da error claro.
- La muestra fija es estable con películas nuevas y con otro orden.

**Pendiente de probar en Colab** (necesita dataset y GPU):
- `python -m p2m.clasificador entrenar` y `python -m p2m.evaluar_clasificador` con el modelo base.
- `python -m p2m.evaluar_resena` y `python -m p2m.evaluar_cartel` (re-medir la base, ver punto 1).
- Abrir la demo con el modelo base.

## Re-medición de la base (en Colab, con esta rama)

| Módulo | Antes (muestra anterior) | Ahora (muestra fija) |
|---|---|---|
| Clasificador (F1 macro, test completo) | 0.537 | **0.537** (reentrenado con el código nuevo: idéntico) |
| Reseña (200 carteles): BERTScore / ROUGE-L | 0.104 / 0.142 | **0.102 / 0.140** |
| Cartel (100 películas): CLIPScore generado / real | 27.8 / 27.0 | **27.6 / 27.0** |
| Cartel: F1 de género generado / real | 0.466 / 0.584 | **0.474 / 0.569** |

Los números casi no se mueven: el cambio no altera el nivel de la base. Lo que cambia es que
las referencias ya no se mueven cuando los equipos agregan películas. README y guía ya usan
los números nuevos.

Demo (`04_demo`) probada con el modelo base: géneros, sinopsis y cartel nuevo sin errores.
