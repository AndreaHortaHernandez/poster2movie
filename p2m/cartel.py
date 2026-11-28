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
from PIL import Image

from p2m.estilos import ESTILOS

MODELO_CARTEL = "stabilityai/sdxl-turbo"  # elegido por el equipo 2 (ver RESULTADOS_equipo2.md); "stabilityai/sd-turbo" usa 3.6 GB en vez de 8.4
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
TIPO = torch.float16 if DISPOSITIVO == "cuda" else torch.float32
ANCHO, ALTO = 512, 768  # 2:3, como un cartel real (el protocolo lo pide). A 640 (4:5) el CLIPScore en val es 1.4 puntos mayor, ver RESULTADOS_equipo2.md
PASOS = 2
# Segunda pasada ("hires fix"): se agranda el cartel a 768x1152 (también 2:3) y se redibuja con fuerza baja.
# Las caras pasan de ~40 a ~60 px y ganan detalle (piel, ojos) sin cambiar la composición. Medido en 6 carteles
# el 27 nov: 4.6 s extra por cartel y 9.95 GB de pico en una T4.
HIRES = (768, 1152)
HIRES_FUERZA = 0.4
HIRES_PASOS = 5


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

            imagen = poner_titulo(quitar_letras(imagen), titulo, generos)
        return imagen

    def mejorar(self, imagen, generos: list[str], resena: str, semilla: int = 0):
        """Segunda pasada a 768x1152 con el mismo modelo (img2img que comparte los pesos, sin memoria extra)."""
        if getattr(self, "_i2i", None) is None:
            from diffusers import AutoPipelineForImage2Image

            self._i2i = AutoPipelineForImage2Image.from_pipe(self.pipeline)
            self._i2i.set_progress_bar_config(disable=True)
        grande = imagen.convert("RGB").resize(HIRES, Image.LANCZOS)
        generador = torch.Generator(DISPOSITIVO).manual_seed(semilla)
        def correr():
            return self._i2i(prompt=instruccion(generos, resena), image=grande, strength=HIRES_FUERZA,
                             num_inference_steps=HIRES_PASOS, guidance_scale=0.0, generator=generador).images[0]

        # El decodificador del VAE a 768x1152 es lo que más memoria pide. Con los tres modelos de la demo
        # en una T4 de 15 GB puede faltar poco: se libera la caché y, solo si aun así no cabe, se
        # decodifica por mosaicos (menos memoria, diferencia visual mínima).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return correr()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._i2i.vae.enable_tiling()
            generador.manual_seed(semilla)
            return correr()
