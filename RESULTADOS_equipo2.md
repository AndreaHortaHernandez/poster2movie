# Resultados del equipo 2: generación del cartel (módulo D)

Rama: `equipo2-cartel`. Líder: Diego López Cano.

> **Estado de los números.** Las elecciones de modelo, estilos y tamaño se hicieron en la muestra de
> test (100 y 300 películas) antes de que existiera el split de validación, así que esas tablas
> (sección 6) son **provisionales**. La versión final se midió **una vez en las 500 películas oficiales de
> test**, con el sistema base en las mismas 500 y bootstrap por película (sección 2). Los experimentos
> posteriores (filtro de anatomía, segunda pasada, modelos alternativos) se hicieron mirando carteles de
> la muestra de val y se declaran como tales.

## 1. Qué cambia en el sistema

| | Base | Equipo 2 (final) |
|---|---|---|
| Modelo | SD-Turbo (3.6 GB), 512 × 768 | **SDXL-Turbo** (8.4 GB), 2 pasos, guidance 0, **512 × 768 (2:3)** |
| Prompt | géneros + primera oración | géneros + **estilos visuales de los 2 primeros géneros** + primera oración (≤ 25 palabras) |
| Selección | una semilla | **primera semilla (hasta 6) sin defectos de anatomía** (`p2m/anatomia.py`); sin selección por CLIPScore |
| Segunda pasada | no | **768 × 1152 (2:3), img2img con fuerza 0.4**, solo se acepta si no sube el conteo de defectos (`Cartelista.mejorar`) |
| Título | no | opcional: borra letras falsas en los bordes y escribe el título (`p2m/cartel_titulo.py`) |

Archivos: `p2m/cartel.py`, `p2m/estilos.py`, `p2m/anatomia.py` (filtro), `p2m/mejor_de_n.py` (selección
y segunda pasada), `p2m/cartel_titulo.py`, `p2m/encuesta.py` y `p2m/encuesta_es.py` (encuesta ciega),
`p2m/evaluar_cartel_val.py` (medición con segundos por cartel, memoria pico y F1 por película).

Uso: `Cartelista().generar(generos, resena)` devuelve el cartel sin texto (así se evalúa). Con
`titulo="..."` devuelve el cartel con título. Para activar filtro y segunda pasada:

```python
from p2m import mejor_de_n
mejor_de_n.activar(n=6, anatomia=True, max_caras=2, usar_clip=False, hires=True)
```

## 2. Medición final en test (500 películas oficiales)

*(Mismas 500 películas para el base y para la versión final. Diferencias con IC95 % por bootstrap de
10 000 remuestreos sobre las diferencias por película.)*

| Medida (500 películas de test) | Base (SD-Turbo) | Final (SDXL-Turbo + filtro + 2.ª pasada) | Final − base, IC95 % |
|---|---|---|---|
| CLIPScore | 27.415 | 26.885 | **−0.531 [−0.796, −0.267]** |
| F1 de género (juez del evaluador) | 0.506 | 0.576 | **+0.071 [+0.046, +0.095]** |
| Segundos por cartel (T4) | 0.61 | 7.08 | — |
| Memoria pico de GPU | 3.57 GB | 9.95 GB | — |

Final contra carteles reales: CLIPScore −0.605 [−1.002, −0.214] y F1 0.000 [−0.030, +0.030]. Es decir, en F1 el
final **empata con los carteles reales**, mientras que el base queda −0.070 [−0.100, −0.040] por debajo. En CLIPScore
el base no se distingue de los reales (−0.075 [−0.474, +0.318]) y el final sí queda por debajo. El final frente a
`test_final5`: CLIPScore +0.017 [−0.012, +0.048], F1 −0.001 [−0.004, +0.002] (sin diferencia: la salvaguarda no
cuesta métrica).

Referencia: en esas 500, los carteles reales sacan CLIPScore 27.490 y F1 0.576.

Una corrida previa de la versión final **sin** la salvaguarda de la segunda pasada (`test_final5`) dio
CLIPScore 26.868, F1 0.577, 6.32 s por cartel y 9.95 GB, con 19.4 % de carteles con defectos detectados
sin filtro, 3.8 % tras el filtro y **11.6 % tras la segunda pasada**: el redibujado inventaba a veces una
mano o una cara de más. Por eso se agregó la salvaguarda (hasta 3 reintentos y, si ninguno sirve, se conserva
la imagen sin redibujar) y se midió de nuevo.

Historial de versiones en las mismas 500 de test (para mostrar que no ocultamos intentos):

| Versión | CLIPScore | F1 de género | s por cartel | Memoria pico |
|---|---|---|---|---|
| Base (SD-Turbo) | 27.415 | 0.506 | 0.60 | 3.57 GB |
| v1: mejor de 4 por CLIPScore (ViT-L/14) | 27.685 | 0.576 | 4.87 | 10.07 GB |
| v4: primera semilla sin defectos, sin CLIP | 27.080 | 0.575 | 1.79 | 9.20 GB |
| Final sin salvaguarda (`test_final5`) | 26.868 | 0.577 | 6.32 | 9.95 GB |
| **Final (`test_final6`)** | 26.885 | 0.576 | 7.08 | 9.95 GB |

La v1 tenía el mejor CLIPScore, pero elegir por CLIPScore favorece carteles recargados y con sujetos repetidos
(caras apiladas, jinetes fusionados). Lo descubrimos al revisar los carteles de la encuesta 3 y la
quitamos aunque bajó la métrica: **el CLIPScore premia justo el defecto que queremos evitar.**

## 2.1 Filtro de anatomía

Cuenta manos y caras con MediaPipe (HandLandmarker y FaceDetector) y descarta la semilla si hay manos de más
para las caras detectadas, más de 2 caras o caras apiladas. En 500 de test: 19.4 % de carteles con algún defecto detectado sin filtro, 3.8 % con el filtro (primera semilla limpia de hasta 6) y 3.6 % tras la segunda pasada con salvaguarda. En 3.8 % de las películas la segunda pasada no logró una versión sin empeorar el conteo y se conservó la imagen sin redibujar (ampliada a 768×1152). Es ciego a lo que no cuenta: caras
borrosas o mal dibujadas, cuerpos fusionados, armas deformes y manos que no detecta. Esos defectos se
revisaron a mano en los carteles mostrados (sección 5).

## 2.2 Encuesta ciega (E12)

*Pendiente: la encuesta está en curso con los compañeros del equipo; los resultados se agregarán aquí (con el reporte honesto de cualquier criterio donde gane el base).*

## 6. Recorrido por rol (experimentos de la fase 1, provisionales)

**Rol 2, estilos A (8 géneros).** Probó tres versiones de frases de estilo en 300 películas:
equilibrados (27.126 / 0.5112), intensos (26.535 / 0.5317) y cinematográficos (26.849 / 0.5485),
contra el base en 300 (27.342 / 0.4976). Se usó la familia de frases de Rol 2 para Comedy, Animation,
Family, Fantasy, Adventure, Action, Science Fiction y Mystery. Su evaluación "final" coincidía con
el base porque se perdió el parche al reiniciar la sesión; se declaró inválida y se sustituyó por la
medición integrada.

**Rol 3, estilos B (9 géneros).** Cinco versiones en 300 películas: sin estilos 27.355 / 0.5046, v1
26.688 / 0.5246, v2 26.757 / 0.5149, v3 26.096 / 0.5085, v4 26.818 / 0.5296. La v4 en las 100
oficiales dio 27.302 / 0.5355: sube el F1 y baja el CLIPScore del base. Cubre Thriller, Horror,
Crime, War, Western, Drama, Romance, History y Documentary.

**Rol 4, modelo y pasos.** Ocho combinaciones de modelo, pasos y tamaño en las 100 películas de
test. Con SD-Turbo a 768, de 1 a 4 pasos el resultado es casi plano (CLIPScore 27.4 a 27.6, F1 0.470 a
0.494). SDXL-Turbo sube el CLIPScore: 28.15 / 0.498 con 2 pasos a 768, 28.23 / 0.488 con 4 pasos y
29.30 / 0.489 a 512 × 512. Tiempos por cartel: SD-Turbo 0.5 s (3.6 GB de GPU) contra SDXL-Turbo 1.2 s
(8.4 GB). Su compañero eligió a ojo SD-Turbo con 4 pasos; en las láminas SDXL se veía más limpio y
SD-Turbo con 4 pasos mostraba letras sin sentido. La decisión quedó condicionada a que SDXL cupiera en
la T4 con el resto de la demo (se midió en la integración, ver sección 4). Su tabla no probó 640 con
SD-Turbo; lo medimos nosotros.

**Rol 5, datos para LoRA.** Armó un conjunto de 2,992 carteles del split de train (17 géneros, 176 por
género principal) con filtros de integridad de archivo, dimensión mínima, proporción vertical y
imágenes casi en blanco, más 8 tandas de revisión visual de 40 carteles (4 imágenes excluidas).
Limitación que él mismo señaló: los carteles originales miden unos 384 px de ancho y se ampliaron a
512 para entrenar. El Rol 6 descartó además las filas con sinopsis de menos de 5 palabras (89 filas).

**Rol 6, entrenamiento LoRA sobre SD-Turbo.** Resultado negativo. En las mismas 100 películas y con el
mismo juez: base 27.614 / 0.476; LoRA final (1000 pasos) 21.118 / 0.296; checkpoint 750 21.004 / 0.289;
checkpoint 500 21.233 / 0.299. El modelo ya estaba dañado desde los 500 pasos, así que no es un exceso
de entrenamiento. Nuestra hipótesis, sin aislar, es que un ajuste fino normal rompe la destilación de
los modelos Turbo. Detalle en `entregas/rol6_entrenamiento/nota_lider_checkpoints_rol6.md`. La versión
final no usa LoRA.

**Rol 7, título.** `quitar_letras` (easyocr + inpainting de OpenCV + degradado oscuro) limpia las
letras que el modelo inventa, y `poner_titulo` escribe el título con una Google Font por género,
color crema, sombra suave y ajuste automático de tamaño. Se integró con dos cambios mínimos: carpeta
de fuentes temporal por defecto y descarga automática al primer uso.

## 3. Experimentos que no funcionaron (resultados negativos)

| Intento | Resultado | Decisión |
|---|---|---|
| SDXL-Lightning 4 pasos (val) | CLIPScore 26.197, F1 0.584, **76 % de carteles con defectos**, 4.39 s, 10.90 GB, contra SDXL-Turbo 768 sin filtro: 26.708 / 0.585 / 12 % / 1.24 s / 8.44 GB | descartado |
| SDXL base 28 pasos, 768 × 1152, negative prompt (9 películas difíciles) | carteles de espaldas, más texto falso grande, se aleja de la sinopsis. No se midió tiempo ni memoria | descartado |
| SD 3.5 Medium sin T5 (7 películas difíciles) | cabe en T4 (8.97 GB) pero **36.9 s por cartel**; texto falso enorme, personajes ausentes, estilo de póster impreso. No se afinó el prompt para este modelo | descartado; sin probar FLUX por tamaño |
| LoRA sobre SD-Turbo (Rol 6) | CLIPScore 27.6 → 21.1 y F1 0.476 → 0.296 en 100 películas; dañado desde 500 pasos | descartado; hipótesis (sin aislar): el ajuste fino rompe la destilación Turbo |
| Mejor de N por CLIPScore | sube CLIPScore pero elige carteles recargados y con sujetos repetidos | reemplazado por filtro sin CLIP |
| Detector de objetos (EfficientDet) para repetidos | marcó 14 de 20 carteles finales y no vio defectos reales | desactivado (queda en el código) |
| Retoque de caras estilo ADetailer (recorte + img2img + pegado) | tomó a un perro por cara y dibujó una niña; cambió identidades; parches visibles | descartado y borrado |
| Formatos cuadrados recortados a 2:3 (512², 768²) | 3 y 5 carteles con defectos contra 2 en 512 × 768; el recorte corta cabezas | se queda 512 × 768 |
| Franja oscura arriba contra texto falso | cortaba cabezas y no tapaba el texto | solo franja inferior |
| Inpaint de texto en todo el cartel | manchas en forma de rombo y parches rectangulares sobre el dibujo | solo en los bordes |

## 4. Decisiones de la integración

- **2:3.** El protocolo pide proporción de cartel real. Antes se usaba 512 × 640 (4:5), que da 1.4 puntos
  más de CLIPScore en val, pero no es 2:3.
- **SDXL-Turbo.** Mejor calidad y mejor CLIPScore que SD-Turbo. Con clasificador, Qwen3-VL-2B y título
  cargados a la vez, la demo usó 11.9 GB y un pico de 13.2 GB en una T4 de ~15 GB **sin la segunda
  pasada**. La segunda pasada comparte los pesos con el modelo base, pero **no se ha medido el pico de la
  demo completa con ella** (pendiente, sección 5).
- **Estilos A + B**, dos por cartel (`GENEROS_CON_ESTILO = 2`).
- **Límite de CLIP.** El texto se trunca a 77 tokens: 44 de 300 prompts lo pasan y pierden en promedio
  4.1 tokens, casi siempre el final de la oración.

## 5. Limitaciones

- **Caras pequeñas.** Una cara de ~40 px no tiene detalle que dibujar a 512 × 768. La segunda pasada
  ayuda (piel, ojos), pero no corrige una cara mal formada de fondo.
- **El filtro de anatomía solo cuenta.** No ve caras mal dibujadas, cuerpos fusionados, armas deformes
  ni manos que no detecta. En la revisión visual de 20 carteles finales automáticos quedaron dos
  defectos: *200 MPH* (segundo hombre con cara parecida y pistola de doble cañón) e *Inhabited Island*
  (una espada extra).
- **Selección manual en las muestras.** Los 20 carteles de demostración se eligieron a mano entre
  varias semillas; **no son evidencia de la mejora**. La evidencia son la medición de test y la encuesta
  ciega, que usan solo lo que el sistema produce solo.
- **El CLIPScore baja** frente al base, y el F1 sube. Ninguna de las dos métricas ve por sí sola si el
  cartel "se ve bien".
- **Costo.** En test (generar solo el cartel): 7.08 s por cartel y 9.95 GB de pico de GPU (el base: 0.61 s y 3.57 GB).
  **Demo completa en una T4 gratuita** (clasificador + Qwen3-VL-2B + SDXL-Turbo, filtro y segunda pasada, título con OCR
  en CPU): en 5 corridas seguidas salieron los 5 carteles (768×1152), con **pico asignado de 14.69 GB y reservado de
  15.25 GB** (la T4 ofrece 14.56 GiB ≈ 15.6 GB) y ~48 s por cartel de punta a punta (la primera vez, 67 s por la carga).
  Cabe, pero con ~1 GB de margen y con `expandable_segments`; la primera versión sin esa protección se quedó sin memoria
  en la segunda corrida (OOM en el decodificador del VAE a 768×1152). `cartel.mejorar` libera caché y, si aun así no
  cabe, decodifica por mosaicos. Si falla en la máquina de alguien, `P2M_SIN_HIRES=1` apaga la segunda pasada
  (menos memoria y tiempo, más defectos).
- **Independencia.** El juez de género del F1 y el CLIPScore son los del evaluador; las cifras de la sección
  2 usan la misma muestra de test que el resto del grupo.
- Los experimentos de modelo alternativo (Lightning, SDXL base, SD 3.5) se hicieron con un prompt
  pensado para SDXL-Turbo; un prompt afinado podría cambiar esas conclusiones.

## 7. Cómo reproducir

```
git checkout equipo2-cartel
pip install -r requirements.txt
python - <<'PY'
from p2m import cartel, mejor_de_n
from p2m import evaluar_cartel_val as v
mejor_de_n.activar(n=6, anatomia=True, max_caras=2, usar_clip=False, hires=True)
v.evaluar_test("stabilityai/sdxl-turbo", "test_final6", 500)
PY
```

Para el base: `cartel.instruccion = encuesta._instruccion_base` y
`v.evaluar_test("stabilityai/sd-turbo", "test_base", 500)`. Los resultados quedan en
`resultados/cartel/<nombre>_metricas.csv`, `_ejemplos.csv`, `_extra.csv` y `_lamina.jpg`.
