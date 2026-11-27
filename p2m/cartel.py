"""
Módulo 3 — Cartel: géneros + reseña -> cartel nuevo.

Versión BASE (la que mejora el equipo del cartel):

SD-Turbo (Stable Diffusion 2.1 destilado por Stability AI) genera una imagen a
partir de un texto. Es un modelo de DIFUSIÓN: parte de ruido y, guiado por el
texto (que lee con un Transformer, el codificador de texto de CLIP), lo va
"limpiando" hasta formar una imagen. La versión Turbo lo hace en 1-4 pasos en
vez de 50, así que tarda ~1 s por cartel en una T4.

El texto se arma con los géneros y la primera oración de la reseña. El
codificador de texto solo lee ~77 tokens, por eso no se le pasa la reseña
completa.

Ideas para mejorarlo (equipo del cartel): mejor instrucción (estilo por
género), más pasos, un modelo más grande (SDXL-Turbo), LoRA con carteles
reales para que se vean más "de cine", poner el título.

Interfaz que usa la demo:
    cartelista = Cartelista()
    cartelista.generar(["Horror"], "A group of friends ...")  ->  imagen PIL de 512 x 640
"""

import re

import torch

from p2m.estilos import ESTILOS

MODELO_CARTEL = "stabilityai/sdxl-turbo"  # elegido por el equipo 2 (ver RESULTADOS_equipo2.md); "stabilityai/sd-turbo" usa 3.6 GB en vez de 8.4
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
TIPO = torch.float16 if DISPOSITIVO == "cuda" else torch.float32
ANCHO, ALTO = 512, 640  # 4:5. SD-Turbo se entrenó a 512x512: a 768 de alto repite sujetos y deforma caras
PASOS = 2


GENEROS_CON_ESTILO = 2   # cuántos de los primeros géneros aportan su estilo al prompt (se ajusta en val)


def instruccion(generos: list[str], resena: str) -> str:
    """Prompt = tipo de cartel + estilo de cada género + primera oración de la reseña.

    Los estilos van ANTES de la sinopsis y el codificador de texto corta en 77 tokens, así que
    cada estilo es corto (~10 palabras) y la oración se recorta a 25 palabras."""
    primera_oracion = re.split(r"(?<=[.!?])\s", resena.strip(), maxsplit=1)[0]
    primera_oracion = " ".join(primera_oracion.split()[:25])
    tipo = "/".join(g.lower() for g in generos[:2]) or "feature"
    estilos = [ESTILOS[g] for g in generos[:GENEROS_CON_ESTILO] if g in ESTILOS]
    return ", ".join([f"Cinematic {tipo} movie poster"] + estilos + ["professional key art", "no text"]) \
        + ": " + primera_oracion


class Cartelista:
    def __init__(self, modelo: str = MODELO_CARTEL):
        from diffusers import AutoPipelineForText2Image

        self.pipeline = AutoPipelineForText2Image.from_pretrained(modelo, torch_dtype=TIPO).to(DISPOSITIVO)
        self.pipeline.set_progress_bar_config(disable=True)

    def generar(self, generos: list[str], resena: str, semilla: int = 0, titulo: str | None = None):
        """Cartel nuevo. Si se da `titulo`, se borran las letras falsas que inventa el modelo y se
        escribe el título (Rol 7). Sin título devuelve el cartel tal cual: así se evalúa, porque
        el CLIPScore y el clasificador miden la imagen sin texto encima."""
        generador = torch.Generator(DISPOSITIVO).manual_seed(semilla)
        imagen = self.pipeline(
            prompt=instruccion(generos, resena),
            width=ANCHO, height=ALTO,
            num_inference_steps=PASOS,
            guidance_scale=0.0,  # SD-Turbo se entrenó sin guía libre de clasificador
            generator=generador,
        ).images[0]
        if titulo:
            from p2m.cartel_titulo import poner_titulo, quitar_letras

            imagen = poner_titulo(quitar_letras(imagen, franja_sup=0.10), titulo, generos)
        return imagen
