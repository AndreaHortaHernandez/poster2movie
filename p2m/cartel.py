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
    cartelista.generar(["Horror"], "A group of friends ...")  ->  imagen PIL de 512 x 768
"""

import re

import torch

MODELO_CARTEL = "stabilityai/sd-turbo"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
TIPO = torch.float16 if DISPOSITIVO == "cuda" else torch.float32
ANCHO, ALTO = 512, 768  # proporción 2:3 de un cartel
PASOS = 2


def instruccion(generos: list[str], resena: str) -> str:
    """El estilo va primero y la oración se recorta a 35 palabras: el codificador
    de texto corta en 77 tokens y lo que quede después se ignora."""
    primera_oracion = re.split(r"(?<=[.!?])\s", resena.strip(), maxsplit=1)[0]
    primera_oracion = " ".join(primera_oracion.split()[:35])
    tipo = "/".join(g.lower() for g in generos[:2]) or "feature"
    return (f"Cinematic {tipo} movie poster, professional key art, dramatic lighting, "
            f"no text: {primera_oracion}")


class Cartelista:
    def __init__(self, modelo: str = MODELO_CARTEL):
        from diffusers import AutoPipelineForText2Image

        self.pipeline = AutoPipelineForText2Image.from_pretrained(modelo, torch_dtype=TIPO).to(DISPOSITIVO)
        self.pipeline.set_progress_bar_config(disable=True)

    def generar(self, generos: list[str], resena: str, semilla: int = 0):
        generador = torch.Generator(DISPOSITIVO).manual_seed(semilla)
        return self.pipeline(
            prompt=instruccion(generos, resena),
            width=ANCHO, height=ALTO,
            num_inference_steps=PASOS,
            guidance_scale=0.0,  # SD-Turbo se entrenó sin guía libre de clasificador
            generator=generador,
        ).images[0]
