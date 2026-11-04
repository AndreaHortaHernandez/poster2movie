# Guía para los equipos — mejorar el sistema base

El sistema base ya funciona de principio a fin (cartel → géneros + sinopsis + cartel nuevo).
Cada equipo se encarga de **mejorar una parte** y demostrarlo con números.

| Rol | Archivo principal | Métrica principal | Notebook |
|---|---|---|---|
| A. Datos y reporte | `p2m/construir_dataset.py` | Calidad del dataset (ver abajo) | `00_construir_dataset` |
| B. Clasificador | `p2m/clasificador.py` | **F1 macro** en test | `01_clasificador` |
| C. Reseña (PLN) | `p2m/resena.py` | **BERTScore** en test | `02_resena` |
| D. Cartel | `p2m/cartel.py` | **CLIPScore** y **F1 de género** en test | `03_cartel` |

## Calendario

| Fechas | Qué |
|---|---|
| 12 nov | Reciben el sistema base con sus métricas; cada equipo elige su rol (por votación) |
| 12 – 30 nov | Mejoran su parte |
| 30 nov | Fecha límite para subir su versión final y sus métricas |
| 1 – 5 dic | Integración, evaluación final, reporte y presentación (entrega 5 dic) |

## Reglas para todos

1. **No cambien el split de test** (`config.split_de`) ni la muestra fija de los evaluadores. Si lo cambian,
   sus números ya no se pueden comparar con la base.
2. **No cambien la interfaz** de su módulo (la clase y el método que usa la demo:
   `Clasificador.predecir`, `Resenador.generar`, `Cartelista.generar`). Por dentro hagan lo que quieran.
3. Cada experimento se evalúa con un nombre distinto (`--nombre equipoX_idea`) para no sobrescribir
   los resultados de la base.
4. Trabajen en una **rama** (`git checkout -b clasificador-lora`) y abran un Pull Request; no suban
   directo a `main`.
5. Guarden en su reporte una tabla **base vs. cada intento**, aunque un intento no haya mejorado:
   explicar por qué no funcionó también cuenta.
6. Todo se puede correr gratis en Colab (GPU T4). El dataset (`poster2movie_datos`) se les comparte
   **solo para leer**: agréguenlo a *Mi unidad* como acceso directo. Sus modelos y resultados van en una
   carpeta suya: en la primera celda de cada notebook quiten el `#` de la línea `P2M_SALIDAS` y pongan
   su equipo. El código usa sus archivos si existen y, si no, los del sistema base.

---

## A. Datos y reporte

**Objetivo:** que el dataset esté limpio, documentado, y armar el reporte final del proyecto.

Tareas:
- **Películas recientes (2023 en adelante).** El dataset base casi no tiene películas nuevas (la mitad
  es de 1988-2009 y llega hasta 2023), y es lo primero que alguien va a probar en la demo. Agregar
  muchas con `scripts/descargar_tmdb.py ... --desde 2023-01-01`, en la carpeta del equipo al que
  le toca cada género (la película va al equipo de su primer género).
- **Lista de películas específicas** (la da la coordinación del proyecto): revisar una por una si ya
  están en el dataset (base o equipos) y, si no, agregarlas. Son películas que pueden preguntar en
  la presentación.
- Revisar los problemas conocidos y arreglarlos con los equipos: películas incompletas
  (`scripts/quitar_incompletas.py`), lotes que se sobrescribieron, carteles rotos.
- Análisis del dataset para el reporte: películas por género y por split, desbalance, idiomas, años,
  largo de sinopsis, cuántas tienen franquicia. Gráficas.
- Revisar a mano una muestra de carteles por género: ¿hay carteles que no son carteles (fotos, logos,
  imágenes en blanco)? ¿sinopsis en otro idioma? Proponer y medir filtros.
- Si se agregan películas nuevas, volver a correr `00_construir_dataset` (el split no cambia para las
  que ya estaban, porque se decide por hash de la franquicia).
- **Reporte final:** juntar las tablas base vs. mejorado de los otros tres equipos.

Métricas de calidad: % de películas descartadas y por qué, carteles problemáticos encontrados en la
revisión manual, balance de géneros antes/después.

## B. Clasificador (cartel → géneros)

**Base:** CLIP ViT-B/32 congelado + una capa lineal + umbral por género.

**Punto de partida (test, 6,237 películas): F1 macro = 0.537** (trivial: 0.065), F1 micro 0.598,
mAP 0.532, acierto exacto 13.9 %.

| Mejor F1 | | Peor F1 | |
|---|---|---|---|
| Animación | 0.87 | Misterio | 0.27 |
| Drama | 0.72 | Historia | 0.33 |
| Comedia | 0.68 | Fantasía | 0.36 |
| Terror | 0.67 | Crimen | 0.43 |

Los géneros débiles son los que en el cartel "se ven" igual que otros (Misterio ↔ Thriller,
Historia ↔ Drama/Guerra, Crimen ↔ Thriller): ahí está el margen de mejora.

Ideas de mejora (de menos a más trabajo):
1. Un CLIP más grande (`openai/clip-vit-large-patch14`) o SigLIP (`google/siglip-base-patch16-224`):
   solo cambia `MODELO_CLIP` y recalcula los vectores.
2. Una red de 2 capas (MLP) en lugar de una lineal.
3. Combinar con el texto: "zero-shot" de CLIP con frases tipo "a horror movie poster".
4. Afinar las últimas capas de CLIP (fine-tuning) con *data augmentation*, como en la práctica de frutas.

Medir: `python -m p2m.evaluar_clasificador --modelo <ruta> --nombre equipoX_idea`.
Fijarse sobre todo en los géneros con F1 bajo (`<nombre>_por_genero.csv`).

## C. Reseña (cartel → sinopsis)

**Base:** Qwen3-VL-2B-Instruct sin entrenar, con una instrucción.

**Punto de partida (200 carteles de test, géneros reales): BERTScore = 0.104**, ROUGE-L 0.142.
Las sinopsis generadas tienen 67 palabras en promedio contra 55 de las reales (son más largas de
lo que deberían). Lo que se ve en los ejemplos (`base_ejemplos.csv`):
- El modelo **lee el título** del cartel y, si conoce la película, acierta el tema (*The Last
  Emperor* → dinastía Qing) pero **inventa detalles** (*Alive* en el Himalaya en 1997, cuando es en
  los Andes; *Super Mario Bros.* "animada", cuando es con actores).
- En películas poco conocidas escribe una sinopsis genérica del género ("a web of secrets that
  threaten to unravel...").

Ideas de mejora:
1. Mejorar la instrucción (ejemplos de sinopsis reales en el prompt, pedir cierto largo).
2. **Afinar con LoRA** sobre los pares cartel → `overview` del split train (con `peft`, como el latín
   de la práctica 1 del tercer parcial).
3. Otro modelo (SmolVLM2-2.2B, Qwen2.5-VL-3B).
4. Generar en español (y decidir cómo medirlo).

Medir: `python -m p2m.evaluar_resena --modelo <ruta o id> --nombre equipoX_idea`. También con
`--generos predichos` para ver el sistema completo. Revisar los textos en `<nombre>_ejemplos.csv`:
ojo con películas famosas donde el modelo "recuerda" la sinopsis por el título del cartel.

## D. Cartel (géneros + sinopsis → cartel nuevo)

**Base:** SD-Turbo en 2 pasos, 512 × 768.

**Punto de partida (100 películas de test):**

| | Generado | Cartel real |
|---|---|---|
| CLIPScore (¿corresponde a la sinopsis?) | **27.8** | 27.0 |
| F1 de género (¿el clasificador reconoce el género?) | **0.466** | 0.584 |

- El CLIPScore del generado sale **igual o mayor** que el del real porque se genera directo a partir
  de la sinopsis; los carteles reales son más "artísticos" y no ilustran la trama literal. Por eso la
  métrica que más dice es la de **género**: el generado logra el 80 % de lo que logra un cartel real.
- En la lámina (`base_lamina.jpg`) se ven carteles creíbles y de época, pero con **letras sin
  sentido** ("CCIAILIE", "EDEGEEP") aunque la instrucción dice *no text*: SD-Turbo no sabe escribir
  y, como se usa sin guía (`guidance_scale=0`), tampoco acepta *negative prompt*.

Ideas de mejora:
1. Mejor instrucción: estilo distinto por género (colores, iluminación), más pasos (1-4).
2. Un modelo más grande: `stabilityai/sdxl-turbo`.
3. **LoRA de estilo "cartel de cine"** entrenado con carteles reales del split train (si hace falta
   más resolución, bajar de Hugging Face solo los carteles que usen, en tamaño original).
4. Quitar las letras falsas y poner el título de verdad (por ejemplo, dibujándolo encima con PIL).

Medir: `python -m p2m.evaluar_cartel --modelo <id> --nombre equipoX_idea`. Comparar contra el cartel
real (`CLIPScore_real`, `F1_genero_real`) y revisar la lámina `<nombre>_lamina.jpg`.
