"""
Módulo 2 — Reseña: cartel (+ géneros) -> sinopsis de la película.

Un modelo de visión y lenguaje (Qwen3-VL-2B-Instruct, Transformer que recibe
una imagen y texto y genera texto) mira el cartel y, con los géneros como
pista, escribe una sinopsis de 2-3 oraciones al estilo de TMDB. Afinado con
LoRA sobre los pares cartel -> overview del split train (equipo de PLN),
publicado en Ddarwin08/poster2movie-resena-equipo4: mejora el ROUGE-L y el
BERTScore del zero-shot original, aunque sigue generando sinopsis más cortas
(~30 palabras) que las reales (~58). Se probaron 4 arreglos para el largo
(forzar más tokens, generar en dos pasadas, beam search, reentrenar con datos
filtrados por largo) y los 4 empeoraron el BERTScore -- el texto de más
siempre resultó en alucinar detalles de trama equivocados o en repeticion
degenerada, nunca en más contenido real. Detalle completo en
notebooks/02_resena.ipynb.

Se genera en inglés porque las sinopsis del dataset (las referencias contra
las que se mide) están en inglés.

Interfaz que usan la demo y el módulo de cartel:
    resenador = Resenador()
    resenador.generar(imagen_pil, generos=["Horror", "Thriller"])  ->  "A group of friends ..."
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODELO_RESENA = "Ddarwin08/poster2movie-resena-equipo4"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
# La T4 de Colab no tiene bfloat16 nativo (es lento); float16 sí.
TIPO = torch.float16 if DISPOSITIVO == "cuda" else torch.float32
MAX_TOKENS = 120

INSTRUCCION = (
    "This is the poster of a movie{generos}. Write only the movie's synopsis, in the style of a "
    "TMDB overview: who the main character(s) are, what happens to them, and what is at stake. "
    "Base it only on what the poster shows (characters, setting, mood, any visible text) and the "
    "genre(s) given above — even if you recognize the title, do not use any other knowledge you "
    "may have about this specific movie, and do not describe the poster itself. Write exactly 2 "
    "sentences, 45 to 60 words total.\n\n"
    "Example style (a different, made-up movie, for reference only):\n"
    "\"A small-town detective is pulled back into the case that ended her career when a copycat "
    "killer starts recreating the crime scenes she investigated years ago. As the body count "
    "rises, she must decide whether catching him is worth confronting the truth she buried.\""
)


def construir_instruccion(generos: list[str] | None) -> str:
    pista = f" whose genres are {', '.join(generos)}" if generos else ""
    return INSTRUCCION.format(generos=pista)


class Resenador:
    def __init__(self, modelo: str = MODELO_RESENA):
        self.procesador = AutoProcessor.from_pretrained(modelo)
        self.modelo = AutoModelForImageTextToText.from_pretrained(modelo, dtype=TIPO).to(DISPOSITIVO).eval()

    def instruccion(self, generos: list[str] | None) -> str:
        return construir_instruccion(generos)

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
