# Equipo 4 — Reseña: resultados del entrenamiento LoRA

## Modelo final

**Hugging Face:** [`GadiGG/poster2movie-resena-equipo4-v2`](https://huggingface.co/GadiGG/poster2movie-resena-equipo4-v2) (público)

Qwen3-VL-2B-Instruct con dos rondas de LoRA ya fusionadas (`merge_and_unload`), así que se carga
igual que el modelo base, sin tocar la interfaz de `Resenador`. `p2m/resena.py` ya apunta aquí por
default, y `Resenador()` ya trae horneados los ajustes de decodificación que dieron el mejor
resultado (`no_repeat_ngram_size=3` + MBR con 8 muestras), así que la demo no necesita ningún
cambio:

```python
from p2m.resena import Resenador
resenador = Resenador()  # ya usa el modelo y la decodificacion finales
```

```bash
python -m p2m.evaluar_resena --cuantos 200 --nombre equipo4_final --no-repetir 3 --mbr 8
```

## Métricas en test (200 carteles, muestra fija, géneros reales)

| Versión | ROUGE-L | BERTScore | Palabras generada / real |
|---|---|---|---|
| Base (README, zero-shot) | 0.140 | 0.102 | 66 / 58 |
| `equipo4_prompt` (prompt mejorado, zero-shot) | 0.134 | 0.106 | 47.3 / 58.3 |
| `equipo4_lora` (ronda 1, LoRA, greedy) | 0.145 | 0.154 | 29.9 / 58.3 |
| **`equipo4_final` (ronda 2 + no-repetir + MBR)** | **0.151** | **0.163** | **38.3 / 58.3** |

**Contra el baseline zero-shot (base), la versión final sube el BERTScore 60 % (0.102 → 0.163) y
el ROUGE-L 8 % (0.140 → 0.151).** La ronda 1 ya había dado la mayor parte de esa mejora, pero
generaba sinopsis muy cortas (~30 palabras vs ~58 reales); la versión final, contra su propia
ronda 1, además sube el largo ~30 % (30 → 38 palabras), aunque todavía se queda por debajo de las
58 palabras reales.

**Nota sobre el MBR:** elige la candidata con mayor BERTScore promedio contra las demás, la misma
métrica con la que después se evalúa. No usa la sinopsis real para elegir, así que no es trampa,
pero sí puede inflar un poco ese número específico. Que el ROUGE-L (una métrica distinta, sin
relación con cómo MBR elige) también mejore ayuda a confirmar que la ganancia es real y no solo un
artefacto de optimizar para la métrica de evaluación.

## Cómo se entrenó

Todo en `notebooks/02_resena.ipynb` (Colab, GPU T4); detalle completo de cada celda y cada
intento ahí mismo.

**Ronda 1** (sección 3): LoRA desde cero sobre Qwen3-VL-2B-Instruct.

| Parámetro | Valor |
|---|---|
| Datos | 3000 pares cartel → `overview` del split **train** (semilla 123) |
| Épocas | 1 (375 pasos), lote efectivo 8, LR 2e-4 (AdamW) |
| LoRA | r = 16, alpha = 32, dropout 0.05, en `q_proj`/`k_proj`/`v_proj`/`o_proj` del modelo de lenguaje |
| Parámetros entrenables | 6,422,528 (0.30 %); torre de visión congelada |
| Tiempo | ~9.3 s/cartel (~7.7 h en total, 3 sesiones de Colab con checkpoint) |

**Ronda 2** (secciones 8-10): continuar el LoRA sobre el modelo ya fusionado de la ronda 1, para
arreglar el largo sin perder lo ya aprendido.

| Parámetro | Valor |
|---|---|
| Datos | 1500 pares, filtrados a overviews de **35-75 palabras** (23,799 de 48,325 caían ahí) — quita el atajo de aprender a cortar corto |
| Épocas | 1 (187 pasos), lote efectivo 8, LR 5e-5 (más baja que la ronda 1) |
| Checkpoint | orden fijo por semilla + índice/paso/optimizador guardados cada 10 pasos, para retomar exacto si Colab se desconecta |

## Intentos que NO funcionaron (documentados en el notebook)

El problema de fondo era el largo (~30 vs ~58 palabras). Se probaron 5 arreglos antes de llegar al
final; los primeros 4 **empeoraron el BERTScore** respecto a `equipo4_lora`:

| Intento | Resultado | Por qué falló |
|---|---|---|
| Forzar `min_new_tokens` | peor | el modelo repetía frases para rellenar |
| Generar en dos pasadas (borrador + expandir) | peor | alucinaba detalles de trama al expandir |
| Beam search (`num_beams=4`) | peor (BERTScore 0.129 en val) | más específico, pero con detalles de trama inventados |
| Reentrenar con datos filtrados (ronda 2, sin más) | peor (BERTScore 0.119 en val) | entró en bucles de repetición ("...forced to work in the camp's laundry..." x9) |
| Ronda 2 + `no_repeat_ngram_size=3` | empate con ronda 1 (quita los bucles) | arregla la repetición, pero sigue alucinando igual que antes |
| **Ronda 2 + no-repetir + MBR(8)** | **mejora limpia** | MBR genera varias sinopsis y se queda con la de consenso; lo que el modelo inventa al azar no se repite entre candidatas y pierde la votación |

Nota de método: los primeros 3 intentos se habían medido contra el número de ronda 1 en **test**
(0.154), pero ellos mismos se midieron en **val**. Al medir los 3 en val parejo contra ronda 1
(0.140 en val), las diferencias eran más chicas de lo que parecían — por eso antes de descartar
un intento conviene medir ambos lados en el mismo split.

## Observaciones

- El modelo ya casi no "recuerda" películas famosas por el título del cartel (p. ej. *The War of
  the Worlds*, *Nineteen Eighty-Four*): se basa más en el cartel y los géneros.
- MBR cuesta ~9x más generación por cartel (greedy + 8 muestras), pero sigue siendo del orden de
  segundos por cartel; no afecta la interfaz ni el tiempo de respuesta de la demo de forma notoria.
- Los CSV de métricas y ejemplos de cada intento están en la carpeta de salidas del equipo en
  Drive: `poster2movie_equipo4/resultados/resena/`.
- Durante las pruebas, el equipo notó que si el modelo recibe carteles de mejor calidad, la
  probabilidad de obtener una reseña más cercana a la trama real parece aumentar un poco —
  razonable, ya que los carteles del dataset se comprimen a 384px para ahorrar espacio (justificado,
  dado el tamaño del dataset completo). No se investigó más a fondo ni se tradujo en ningún cambio,
  solo se deja anotado por si a futuro se quiere explorar.
