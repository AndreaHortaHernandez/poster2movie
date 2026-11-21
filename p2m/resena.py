"""
Módulo 2 — Reseña: cartel (+ géneros) -> sinopsis de la película.

Un modelo de visión y lenguaje (Qwen3-VL-2B-Instruct, Transformer que recibe
una imagen y texto y genera texto) mira el cartel y, con los géneros como
pista, escribe una sinopsis de 2-3 oraciones al estilo de TMDB. Afinado con
LoRA en dos rondas sobre los pares cartel -> overview del split train (equipo
de PLN), publicado en GadiGG/poster2movie-resena-equipo4-v2. La ronda 1 ya
mejoraba el zero-shot pero generaba sinopsis muy cortas (~30 palabras vs ~58
reales); forzar más tokens, dos pasadas y beam search solo empeoraron el
BERTScore (alucinaban detalles de trama). Lo que sí funcionó: reentrenar con
el train filtrado a 35-75 palabras (quita el atajo de aprender a cortar
corto) + generar sin repetir frases (`no_repeat_ngram_size`, evita los bucles
que metio el reentrenamiento) + MBR (Minimum Bayes Risk: generar varias
sinopsis y quedarse con la de consenso, que descarta lo que el modelo
inventa al azar). Resultado en test: ROUGE-L 0.145->0.151, BERTScore
0.154->0.163, largo 29.9->38.3 palabras. Detalle completo (todos los intentos,
los que fallaron y los que no) en notebooks/02_resena.ipynb.

Se genera en inglés porque las sinopsis del dataset (las referencias contra
las que se mide) están en inglés.

Interfaz que usan la demo y el módulo de cartel:
    resenador = Resenador()
    resenador.generar(imagen_pil, generos=["Horror", "Thriller"])  ->  "A group of friends ..."
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODELO_RESENA = "GadiGG/poster2movie-resena-equipo4-v2"
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
    def __init__(self, modelo: str = MODELO_RESENA, mbr: int = 8, **generacion):
        """generacion: opciones extra para generate() (p. ej. num_beams), para experimentar
        desde evaluar_resena; por default ya trae no_repeat_ngram_size=3 (ver mas abajo).

        mbr > 1 (default 8, la validada en el notebook): Minimum Bayes Risk. Ademas de la
        greedy genera `mbr` sinopsis con muestreo y devuelve la de consenso (la de mayor
        BERTScore promedio contra las demas candidatas; la sinopsis real nunca se usa). Lo
        que el modelo inventa al azar no se repite entre candidatas y pierde. mbr=0 para
        greedy simple (p. ej. al comparar contra versiones viejas)."""
        self.procesador = AutoProcessor.from_pretrained(modelo)
        self.modelo = AutoModelForImageTextToText.from_pretrained(modelo, dtype=TIPO).to(DISPOSITIVO).eval()
        self.generacion = {"max_new_tokens": MAX_TOKENS, "do_sample": False,
                            "no_repeat_ngram_size": 3, **generacion}
        self.mbr = mbr
        self._puntuador = None

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
        largo = entrada["input_ids"].shape[1]

        def decodificar(salida):
            return [t.strip() for t in self.procesador.batch_decode(salida[:, largo:], skip_special_tokens=True)]

        candidatas = decodificar(self.modelo.generate(**entrada, **self.generacion))
        if self.mbr <= 1:
            return candidatas[0]

        torch.manual_seed(0)  # mismas muestras en cada corrida, para poder comparar
        muestreo = {**self.generacion, "do_sample": True, "temperature": 0.7, "top_p": 0.9,
                    "num_return_sequences": self.mbr}
        candidatas += decodificar(self.modelo.generate(**entrada, **muestreo))
        return self._consenso(candidatas)

    def _consenso(self, candidatas: list[str]) -> str:
        if self._puntuador is None:
            from bert_score import BERTScorer
            self._puntuador = BERTScorer(lang="en", rescale_with_baseline=True, device=DISPOSITIVO)
        n = len(candidatas)
        pares = [(i, j) for i in range(n) for j in range(i + 1, n)]
        _, _, f1 = self._puntuador.score([candidatas[i] for i, _ in pares], [candidatas[j] for _, j in pares])
        total = [0.0] * n
        for (i, j), valor in zip(pares, f1.tolist()):
            total[i] += valor
            total[j] += valor
        return candidatas[max(range(n), key=total.__getitem__)]
