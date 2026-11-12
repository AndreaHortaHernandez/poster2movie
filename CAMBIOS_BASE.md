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

## 4. Usar el modelo del repo del clasificador (rama `equipo1-migracion`)

**Problema.** El equipo del clasificador trabaja en su propio repo
([11909/Poster2Movie-Clasificador](https://github.com/11909/Poster2Movie-Clasificador)) y
publica cada modelo como release. Su `.pt` trae la llave `vectores` (p. ej.
`["clip_b32__centro"]`), que dice con qué codificadores se calcularon los vectores de entrada.
El `Clasificador` de aquí ignoraba esa llave: siempre creaba un CLIP con `modelo_clip`. Un `.pt`
con otro codificador o con varios (`embeddings = None`) fallaba al cargar o, peor, corría en
silencio con el codificador equivocado. Además `evaluar_clasificador` solo sabía evaluar con
un archivo de vectores precalculado.

**Cambio.**
- `p2m/codificadores.py` (nuevo): copia de la parte de inferencia de
  `clasificador/codificadores.py` del repo del clasificador (commit 1853825), con
  `crear_codificador("clip_b32__centro")`. Con varios `vectores`, los concatena en el orden
  del `.pt`, que es como entrena `clasificador.datos.cargar_split`; si dos variantes usan el
  mismo modelo, se carga una sola vez (memoria de la T4). Un codificador que no esté copiado
  da un error claro. **Al agregar codificadores o variantes allá, hay que traerlos aquí.**
  Las cabezas (`lineal`, `mlp`) ya eran iguales en los dos repos; si agregan otra, se copia a
  `construir_cabeza`.
- `Clasificador(ruta)`: si el `.pt` trae `vectores`, usa esos codificadores; si no, hace
  exactamente lo de antes. `predecir` no cambia.
- `Clasificador()` sin ruta usa el release fijado en `RELEASE_CLASIFICADOR`
  (`p2m/clasificador.py`): `{"tag": "vX.Y.Z", "archivo": "<experimento>.pt", "sha256": ...}`.
  Se descarga una vez del release de GitHub a `~/.cache/poster2movie` (o `P2M_CACHE`) y se
  verifica el sha256 si se da. Sin red y sin caché, o si el sha256 no coincide, avisa y usa
  el clasificador base. Fijado en `v0.1.0` (ver sección 7).
- `evaluar_clasificador`: si el `.pt` trae un archivo de vectores que existe, evalúa como
  siempre (segundos). Si no (varios codificadores o vectores ausentes), codifica cada cartel
  de test con el `Clasificador` completo: mismas películas y métricas, pero tarda minutos.
- No cambia: `ruta_modelo_base()` ni el juez de `evaluar_cartel` (sigue siendo el base) ni
  el `.pt` base. `evaluar_resena --generos predichos` y la demo usan `Clasificador()`, así
  que pasarán al modelo nuevo cuando se fije el release (es el flujo real de la demo).

**Cómo se verificó** (local, sin GPU, con un dataset sintético de 30 carteles):
- El mismo modelo guardado con el formato viejo y con `vectores: ["clip_b32__centro"]` da
  exactamente las mismas probabilidades.
- Con dos `vectores`, la cabeza recibe 1024 = 512 + 512 en orden y CLIP se carga una vez.
- Un codificador no copiado (`siglip_base`) da error claro.
- Release: se descarga una vez (URL `.../releases/download/<tag>/<archivo>`) y luego sale de
  la caché; con sha256 distinto o sin red, aviso y base. `ruta_modelo_base()` no cambia.
- `evaluar_clasificador` con vectores y codificando los carteles da las mismas métricas.

**Pendiente de probar en Colab:** `python -m p2m.evaluar_clasificador` con el base sigue en
F1 macro 0.537 (ese camino no cambió); la demo con un release real; memoria en la T4 cuando
el modelo use varios codificadores.

## 5. Sin `P2M_SALIDAS`, el entrenamiento se perdía al final

**Problema.** Si un equipo no definía `P2M_SALIDAS`, la carpeta de salidas era la del dataset,
que se comparte solo para leer. `python -m p2m.clasificador entrenar` entrenaba completo y
fallaba al guardar el `.pt` con `RuntimeError: ... open file failed with strerror: Operation
not permitted`, un mensaje que no dice qué hacer. Lo mismo pasaba con los vectores y con los
evaluadores, que guardan sus resultados después de minutos de trabajo.

**Cambio.** `config.comprobar_salidas()` crea y borra un archivo de prueba en la carpeta de
salidas. Si no puede, da un `PermissionError` que explica que falta `P2M_SALIDAS` (o que su
ruta no sirve). Se llama al **inicio** de `clasificador entrenar` / `embeddings`,
`evaluar_clasificador`, `evaluar_resena` y `evaluar_cartel`.

**Cómo se verificó.** Con una carpeta donde no se puede escribir, sin y con `P2M_SALIDAS`, se
obtiene el mensaje correspondiente. En una carpeta normal no deja archivos. Las pruebas de la
sección 4 siguen pasando.

## 6. Ensambles y apilamiento del repo del clasificador

**Problema.** El equipo del clasificador definió un `.pt` "combinado" (su contrato 4) para el
ensamble (varias cabezas, una por codificador, con sus probabilidades promediadas) y el
apilamiento (una etapa 2 que corrige las 17 probabilidades). El `Clasificador` de aquí solo
sabía usar **una** cabeza, así que no podía cargar esos modelos. Además, `evaluar_clasificador`
tomaba por defecto los vectores de la base cuando un `.pt` no traía la llave `embeddings`: con
un combinado entraba al camino rápido y fallaba buscando una sola `capa`.

**Cambio.**
- `p2m/combinado.py` (nuevo): copia de la parte de inferencia de `clasificador/combinado.py` del
  repo del clasificador (commit 9761cd8). Promedio con pesos, etapa 2 opcional (con `residual`,
  su salida se suma al logit del promedio) y los umbrales del combinado.
- `Clasificador`: un `.pt` con `vectores` (un experimento) se trata como un combinado de un
  miembro, y un `tipo: combinado` se carga tal cual. Se crea **un codificador por archivo de
  vectores**, compartido entre miembros, y un solo CLIP por modelo. El `.pt` base (sin
  `vectores`) sigue exactamente como antes. `predecir` no cambia.
- `p2m/codificadores.py`: se quita `CodificadorVarios` (lo reemplaza el combinado).
- `evaluar_clasificador`: un combinado siempre se evalúa codificando los carteles de test.

**Cómo se verificó.**
- Con CLIP B/32 real, el mismo `.pt` da **exactamente** las mismas probabilidades (diferencia 0.0)
  en `p2m.clasificador.Clasificador` y en `clasificador.predictor.Clasificador` del repo del
  clasificador, para un experimento normal y para un combinado de dos miembros con pesos 2:1 y
  apilamiento, armado con el código de allá.
- Con un dataset sintético: el combinado da el promedio con pesos y la etapa 2 esperados, se evalúa
  con `evaluar_clasificador`, y las pruebas de la sección 4 siguen pasando.

**Pendiente de probar en Colab:** memoria en la T4 con un ensamble real junto con la reseña y el cartel.

## 7. Integración de prueba con el release `v0.1.0`

**Para qué.** Probar de punta a punta, antes de la entrega, que el repo principal usa los releases
del repo del clasificador: descarga, verificación, caché y carga. `v0.1.0` es la base reproducida
allá (F1 macro en test **0.537**, igual que aquí), así que la demo da los mismos géneros.

**Cambio.** `RELEASE_CLASIFICADOR = {"tag": "v0.1.0", "archivo": "base.pt", "sha256": "99d0b790..."}`.
`Clasificador()` (la demo y `evaluar_resena --generos predichos`) usa ese modelo. El juez de
`evaluar_cartel` sigue siendo el base del dataset (`ruta_modelo_base()`, sin cambios).

**Cómo se verificó** (local, CPU, CLIP real, sin dataset ni base local para que no hubiera respaldo):
- Con la caché vacía, `Clasificador()` descarga `base.pt` del release de GitHub y verifica el sha256.
- Da exactamente las mismas probabilidades que el `base.pt` guardado en git en `releases/v0.1.0/`.
- La segunda vez sale de la caché, sin descargar.

**Para cambiar de modelo:** cuando el equipo del clasificador publique otro release, PR aquí que
fije su `tag`, `archivo` y `sha256` y que copie a `p2m/codificadores.py` / `construir_cabeza` los
codificadores o cabezas nuevos (ver `docs/PAREJA_C.md` del repo del clasificador).

**Pendiente de probar en Colab:** abrir la demo con este release.
