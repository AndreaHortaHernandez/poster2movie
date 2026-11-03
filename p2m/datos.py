"""
Lectura del dataset compacto (el que arma construir_dataset.py).

Todos los módulos leen los datos por aquí, así nadie depende de cómo están
guardados los archivos. La carpeta del dataset es config.carpeta_datos()
(variable de entorno P2M_DATOS; en Colab /content/drive/MyDrive/poster2movie_datos).
"""

import io
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

from p2m import config


def carpeta() -> Path:
    ruta = config.carpeta_datos()
    if not (ruta / "metadata.parquet").exists():
        raise FileNotFoundError(
            f"No encuentro el dataset en '{ruta}'. Corre notebooks/00_construir_dataset "
            "o pon la carpeta en la variable de entorno P2M_DATOS."
        )
    return ruta


def metadata(split: str | None = None) -> pd.DataFrame:
    """Todas las películas sin imagen (id, title, overview, generos, split, ...)."""
    tabla = pd.read_parquet(carpeta() / "metadata.parquet")
    return tabla[tabla["split"] == split].reset_index(drop=True) if split else tabla


def peliculas(split: str | None = None, columnas: list[str] | None = None):
    """Recorre las películas (con imagen) archivo por archivo, sin cargarlas todas a memoria.
    Cada elemento es un dict con las columnas del dataset; 'imagen' son bytes JPEG."""
    if columnas is not None:
        columnas = list(dict.fromkeys([*columnas, "split"]))  # se necesita para filtrar
    for archivo in sorted(carpeta().glob("*.parquet")):
        if archivo.name == "metadata.parquet":
            continue
        tabla = pq.read_table(archivo, columns=columnas)
        if split:
            tabla = tabla.filter(pc.equal(tabla["split"], split))
        yield from tabla.to_pylist()


def abrir_cartel(datos: bytes) -> Image.Image:
    return Image.open(io.BytesIO(datos)).convert("RGB")


def vector_generos(generos: list[str]) -> list[int]:
    """['Drama', 'Romance'] -> [0, 0, ..., 1, ..., 1, ...] en el orden de config.GENEROS."""
    presentes = set(generos)
    return [int(g in presentes) for g in config.GENEROS]
