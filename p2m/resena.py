"""
Módulo 2 — Reseña: cartel (+ géneros) -> sinopsis de la película.

Versión BASE (la que mejora el equipo de PLN):

Un modelo de visión y lenguaje (Qwen3-VL-2B-Instruct, Transformer que recibe
una imagen y texto y genera texto) mira el cartel y, con los géneros como
pista, escribe una sinopsis de 2-3 oraciones al estilo de TMDB. No se entrena:
solo se le da una instrucción (zero-shot).

Se genera en inglés porque las sinopsis del dataset (las referencias contra
las que se mide) están en inglés.

Ideas para mejorarlo (equipo de PLN): afinarlo con LoRA sobre los pares
cartel -> overview del split train, mejorar la instrucción, usar un modelo
más grande, generar en español.

Interfaz que usan la demo y el módulo de cartel:
    resenador = Resenador()
    resenador.generar(imagen_pil, generos=["Horror", "Thriller"])  ->  "A group of friends ..."
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODELO_RESENA = "Qwen/Qwen3-VL-2B-Instruct"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
# La T4 de Colab no tiene bfloat16 nativo (es lento); float16 sí.
TIPO = torch.float16 if DISPOSITIVO == "cuda" else torch.float32
MAX_TOKENS = 120

INSTRUCCION = (
    "This is the poster of a movie{generos}. Write the movie's synopsis in 2 or 3 sentences "
    "(40 to 70 words), in the style of a TMDB overview: who the main characters are, what "
    "happens to them and what is at stake. Do not describe the poster itself."
)


class Resenador:
    def __init__(self, modelo: str = MODELO_RESENA):
        self.procesador = AutoProcessor.from_pretrained(modelo)
        self.modelo = AutoModelForImageTextToText.from_pretrained(modelo, dtype=TIPO).to(DISPOSITIVO).eval()

    def instruccion(self, generos: list[str] | None) -> str:
        pista = f" whose genres are {', '.join(generos)}" if generos else ""
        return INSTRUCCION.format(generos=pista)

    @torch.inference_mode()
    def generar(self, imagen, generos: list[str] | None = None) -> str:
        mensajes = [{"role": "user", "content": [
            {"type": "image", "image": imagen},
            {"type": "text", "text": self.instruccion(generos)},
        ]}]
        entrada = self.procesador.apply_chat_template(
            mensajes, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(DISPOSITIVO)
        salida = self.modelo.generate(**entrada, max_new_tokens=MAX_TOKENS, do_sample=False)
        largo = entrada["input_ids"].shape[1]
        return self.procesador.batch_decode(salida[:, largo:], skip_special_tokens=True)[0].strip()
