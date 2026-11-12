"""
Codificadores del clasificador: cartel -> vector, para los modelos del equipo del clasificador.

Copia de la parte de INFERENCIA de `clasificador/codificadores.py` del repo
11909/Poster2Movie-Clasificador (commit 9761cd8). Debe procesar el cartel EXACTAMENTE
igual que allá: si no, la cabeza recibe vectores distintos a los de su entrenamiento y
predice mal sin dar ningún error. Cuando el equipo del clasificador agregue un
codificador o una variante, hay que traer aquí su clase (y anotar el commit).

Un .pt del repo del clasificador dice qué vectores usa en la llave `vectores`, con
nombres `<codificador>__<variante>`:

    ["clip_b32__centro"]                     un codificador (el de la base)
    ["clip_b32__centro", "clip_l14__centro"] varios: la cabeza recibe sus vectores
                                             CONCATENADOS en ese orden (así entrena
                                             `clasificador.datos.cargar_split`)

Los ensambles y el apilamiento (varias cabezas) están en p2m/combinado.py.

Hoy existe: clip_b32 + centro (CLIP ViT-B/32, recorte central de 224 px).
"""

from functools import lru_cache

import numpy as np
import torch

# id corto -> id en Hugging Face (igual que en el repo del clasificador)
CODIFICADORES = {
    "clip_b32": "openai/clip-vit-base-patch32",
}
VARIANTES = ["centro"]
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=None)
def _cargar_clip(modelo_hf: str):
    """Un solo CLIP en memoria por modelo, aunque lo usen varias variantes (la demo corre en una T4)."""
    from transformers import CLIPModel, CLIPProcessor

    return CLIPProcessor.from_pretrained(modelo_hf), CLIPModel.from_pretrained(modelo_hf).to(DISPOSITIVO).eval()


class Codificador:
    """Interfaz común: todo codificador tiene `nombre`, `modelo_hf`, `dim` y `codificar`."""

    nombre: str
    modelo_hf: str
    dim: int

    def codificar(self, imagenes) -> np.ndarray:
        """Lista de imágenes PIL -> matriz float32 (n, dim)."""
        raise NotImplementedError


class CodificadorCLIP(Codificador):
    """CLIP de OpenAI: vectores de imagen normalizados (igual que `p2m.clasificador.CodificadorCLIP`)."""

    def __init__(self, codificador: str = "clip_b32", variante: str = "centro"):
        if codificador not in CODIFICADORES:
            raise NotImplementedError(f"Codificador '{codificador}': no está en p2m/codificadores.py "
                                      "(tráelo del repo del clasificador).")
        if variante not in VARIANTES:
            raise NotImplementedError(f"Variante '{variante}': no está en p2m/codificadores.py "
                                      "(tráela del repo del clasificador).")
        self.nombre = f"{codificador}__{variante}"
        self.modelo_hf = CODIFICADORES[codificador]
        self.procesador, self.modelo = _cargar_clip(self.modelo_hf)
        self.dim = self.modelo.config.projection_dim

    @torch.inference_mode()
    def codificar(self, imagenes) -> np.ndarray:
        entrada = self.procesador(images=imagenes, return_tensors="pt").to(DISPOSITIVO)
        vectores = self.modelo.get_image_features(**entrada)
        if not torch.is_tensor(vectores):  # transformers >= 5 regresa un objeto
            vectores = vectores.pooler_output
        return torch.nn.functional.normalize(vectores, dim=-1).float().cpu().numpy()


def crear_codificador(nombre_vectores: str) -> Codificador:
    """`clip_b32__centro` -> el codificador que produce esos vectores."""
    codificador, _, variante = nombre_vectores.partition("__")
    if codificador.startswith("clip_"):
        return CodificadorCLIP(codificador, variante or "centro")
    raise NotImplementedError(f"Codificador '{codificador}': no está en p2m/codificadores.py "
                              "(tráelo del repo del clasificador).")
