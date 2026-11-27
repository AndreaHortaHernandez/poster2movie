# Resultados del equipo 2: generación del cartel (módulo D)

Rama: `equipo2-cartel`. Líder: Diego López Cano.

> **Estado de los números.** Las cifras de esta página se midieron en las 100 películas oficiales de
> **test**. La elección de modelo, tamaño y estilos se hizo mirando esas mismas películas (y las
> primeras 300 de la muestra anidada), así que son **provisionales**. Cuando el evaluador tenga
> `--split val`, la elección se repite en val y test se mide una sola vez con la versión final.
> *(Pendiente: actualizar esta sección con esa medición.)*

## 1. Qué cambia en el sistema

| | Base | Equipo 2 |
|---|---|---|
| Modelo | SD-Turbo (3.6 GB) | **SDXL-Turbo** (8.4 GB), 2 pasos, guidance 0 |
| Tamaño | 512 × 768 | **512 × 640** (4:5) |
| Prompt | géneros + primera oración | géneros + **dos estilos visuales por género** + primera oración (≤ 25 palabras) |
| Título | no | **opcional**: borra letras generadas y escribe el título con tipografía según el género |

Archivos: `p2m/cartel.py` (modelo, tamaño, prompt), `p2m/estilos.py` (frases de estilo de los 17
géneros), `p2m/cartel_titulo.py` (borrado de letras y título). `requirements.txt` suma
`opencv-python-headless`, `easyocr` y `matplotlib`.

Formato del prompt:

```
Cinematic {género1}/{género2} movie poster, {estilo1}, {estilo2}, professional key art, no text: {primera oración}
```

`Cartelista().generar(generos, resena)` devuelve la imagen sin texto (así se evalúa).
Con `titulo="..."` devuelve el cartel con título. Para volver al modelo ligero:
`Cartelista("stabilityai/sd-turbo")`.

## 2. Resultado final (100 películas de test)

| Versión | CLIPScore | F1 de género |
|---|---|---|
| Sistema base | 27.614 | 0.476 |
| SD-Turbo + nuestro prompt, 768 | 26.926 | 0.548 |
| SD-Turbo + nuestro prompt, 640 | 27.007 | 0.570 |
| SD-Turbo + nuestro prompt, 512 × 512 | 26.942 | 0.525 |
| **SDXL-Turbo + nuestro prompt, 640 (final)** | **28.005** | **0.540** |
| Carteles reales (referencia) | 26.956 | 0.569 |

La versión final se reprodujo en una corrida limpia
(`python -m p2m.evaluar_cartel --cuantos 100 --nombre eq2_integrado_final`) con el mismo resultado:
28.005 / 0.540. Tarda unos 100 s en una T4.

### Con 300 películas de test (muestra anidada, incluye las 100 oficiales)

| Versión | CLIPScore | F1 de género |
|---|---|---|
| Sistema base (SD-Turbo) | 27.342 | 0.498 |
| **SDXL-Turbo 512 × 640 + estilos (final)** | **27.691** | **0.554** |
| Carteles reales (referencia) | 27.250 | 0.578 |

Diferencia contra el base, película por película (CLIPScore): +0.349, IC95% por bootstrap
[-0.013, +0.716]; mejora en el 54 % de las películas. En las 100 oficiales, la diferencia es +0.390, IC95%
[-0.302, +1.075]. En las 300 el intervalo apenas incluye el cero: **la mejora
de CLIPScore es pequeña y no se puede distinguir con certeza del ruido**. El F1 sube +0.056 (0.498 →
0.554), más claro, pero el evaluador solo guarda el promedio y no el valor por película, así que no
calculamos su intervalo.

Lectura: el prompt con estilos ayuda sobre todo al F1 de género (0.476 → 0.540 en 100, 0.498 → 0.554
en 300). El CLIPScore queda por encima del base y de los carteles reales en promedio, pero con esta
muestra la diferencia contra el base no es concluyente. El SD-Turbo a 640 daba mejor F1 (0.570) pero
peor CLIPScore que el base; ninguna de las dos métricas ve por sí sola si el cartel "se ve bien" (ver
limitaciones).

## 3. Recorrido por rol

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

## 4. Decisiones de la integración

- **768 → 640 de alto.** SD-Turbo se entrenó a 512 × 512; a 768 de alto repite sujetos y deforma
  caras. A 640 (4:5) mejora sin perder el formato de cartel. Aun así no es 2:3.
- **SD-Turbo → SDXL-Turbo.** Mejor calidad visible y mejor CLIPScore. Con SDXL-Turbo, clasificador,
  Qwen3-VL-2B y el paso de título cargados a la vez, la demo usó 11.9 GB y un pico de 13.2 GB en una
  T4 de ~15 GB.
- **Estilos A + B, dos por cartel.** `GENEROS_CON_ESTILO = 2`: se añade el estilo de los dos primeros
  géneros.
- **Límite de CLIP.** El texto se trunca a 77 tokens. En 300 prompts: media de 64.9 tokens, máximo 89,
  44 de 300 pasan el límite y pierden en promedio 4.1 tokens (casi siempre el final de la oración).

## 5. Limitaciones

- SDXL-Turbo necesita 8.4 GB de descarga y deja ~1.5 GB de margen en la T4 con la demo completa. Si
  el módulo de reseñas crece, hay que volver a medir (o usar SD-Turbo).
- Las caras en escenas con varias personas (grupos, multitudes) siguen saliendo imperfectas.
- El tamaño es 4:5, no 2:3 como un cartel real.
- CLIPScore y F1 miden parecido con el texto y con el género, no calidad estética. Las revisiones
  visuales por parte del equipo fueron parte de la decisión.
- Las cifras de la sección 2 son provisionales (ver aviso al inicio).

## 6. Cómo reproducir

```
git checkout equipo2-cartel
pip install -r requirements.txt
python -m p2m.evaluar_cartel --cuantos 100 --nombre eq2_integrado_final
```

Los resultados quedan en `resultados/cartel/eq2_integrado_final_metricas.csv`, `_ejemplos.csv` y
`_lamina.jpg` dentro de la carpeta de salidas del equipo.
