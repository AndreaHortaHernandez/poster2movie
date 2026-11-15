# Equipo 4 — Reseña: resultados del entrenamiento LoRA

## Modelo entrenado

**Hugging Face:** [`Ddarwin08/poster2movie-resena-equipo4`](https://huggingface.co/Ddarwin08/poster2movie-resena-equipo4) (público)

Es Qwen3-VL-2B-Instruct con el adaptador LoRA ya fusionado (`merge_and_unload`), así que se carga
igual que el modelo base, sin tocar la interfaz de `Resenador`:

```python
from p2m.resena import Resenador
resenador = Resenador("Ddarwin08/poster2movie-resena-equipo4")
```

```bash
python -m p2m.evaluar_resena --cuantos 200 --nombre equipo4_lora --modelo Ddarwin08/poster2movie-resena-equipo4
```

## Cómo se entrenó

Notebook `notebooks/02_resena.ipynb`, sección 3, en Colab (GPU T4):

| Parámetro | Valor |
|---|---|
| Datos | 3000 pares cartel → `overview` del split **train** (`CUANTOS_TRAIN = 3000`, semilla 123) |
| Épocas | 1 (375 pasos) |
| Lote efectivo | 8 (gradiente acumulado) |
| Tasa de aprendizaje | 2e-4 (AdamW) |
| LoRA | r = 16, alpha = 32, dropout 0.05, en `q_proj`, `k_proj`, `v_proj`, `o_proj` del modelo de lenguaje |
| Parámetros entrenables | 6,422,528 (0.30 % del modelo); la torre de visión queda congelada |
| Precisión | bfloat16 |
| Tiempo | ~9.3 s por cartel (~7.7 h en total, en 3 sesiones de Colab retomando desde el checkpoint) |

## Métricas en test

200 carteles de la muestra fija de test, con los géneros reales (`evaluar_resena.py`):

| Modelo | ROUGE-L | BERTScore | Palabras generada / real |
|---|---|---|---|
| Base (README) | 0.140 | 0.102 | 66 / 58 |
| `equipo4_prompt` | 0.134 | 0.106 | 47.3 / 58.3 |
| **`equipo4_lora`** | **0.145** | **0.154** | **29.9 / 58.3** |

## Observaciones

- El BERTScore sube de 0.106 a 0.154 (+45 % respecto a `equipo4_prompt`) y el ROUGE-L también mejora.
- Las sinopsis generadas son completas (1-2 oraciones, estilo TMDB), pero **cortas**: ~30 palabras contra
  ~58 de las reales. Es un punto de mejora para un siguiente intento.
- El modelo ya casi no "recuerda" películas famosas por el título del cartel (p. ej. *The War of the
  Worlds*, *Nineteen Eighty-Four*): se basa más en el cartel y los géneros. Por eso falla en esas, pero
  acierta bien el tema en otras (*Female Agents*, *Blind Spot: Hitler's Secretary*).

Los CSV de métricas y ejemplos (`equipo4_prompt_*` y `equipo4_lora_*`) están en la carpeta de salidas del
equipo en Drive: `poster2movie_equipo4/resultados/resena/`.
