"""
Configuración compartida del sistema base Poster2Movie.

Un solo lugar con los géneros, el tamaño de los carteles, el reparto
train / val / test y las rutas, para que el dataset y los tres módulos
(clasificador, reseña, cartel) no se desincronicen.
"""

import hashlib
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# =========================================================
# GÉNEROS: los 17 que se repartieron entre los equipos
# =========================================================
# Music y TV Movie no se usan: se quita esa etiqueta y las películas que solo
# tenían esa se descartan. El orden es fijo: es el orden de las salidas del
# clasificador.
GENEROS = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "History", "Horror", "Mystery", "Romance",
    "Science Fiction", "Thriller", "War", "Western",
]
GENEROS_ES = {
    "Action": "Acción", "Adventure": "Aventura", "Animation": "Animación",
    "Comedy": "Comedia", "Crime": "Crimen", "Documentary": "Documental",
    "Drama": "Drama", "Family": "Familia", "Fantasy": "Fantasía",
    "History": "Historia", "Horror": "Terror", "Mystery": "Misterio",
    "Romance": "Romance", "Science Fiction": "Ciencia ficción",
    "Thriller": "Thriller", "War": "Guerra", "Western": "Western",
}
GENEROS_DESCARTADOS = {"Music", "TV Movie"}

# =========================================================
# DATASET
# =========================================================
DATASET_BASE = "skvarre/movie_posters-100k"  # Hugging Face, ~95 mil filas, 43 GB
ANCHO_CARTEL = 384           # px; el alto se escala igual (los carteles son 2:3 -> 384 x 576)
CALIDAD_JPEG = 85            # ~54 KB por cartel; ~3.3 GB el dataset completo
MIN_PALABRAS_SINOPSIS = 20   # igual que scripts/comun.py

RUTA_COLECCIONES_BASE = RAIZ / "data" / "colecciones_base.csv"
RUTA_DATOS = RAIZ / "datos"  # aquí queda (o se descarga) el dataset compacto


def carpeta_datos() -> Path:
    """Carpeta del dataset compacto (y de los modelos base): la variable de entorno
    P2M_DATOS (en Colab, /content/drive/MyDrive/poster2movie_datos) o RUTA_DATOS."""
    return Path(os.environ.get("P2M_DATOS", RUTA_DATOS))


def carpeta_salidas() -> Path:
    """Dónde se GUARDAN los modelos entrenados (modelos/), los vectores calculados
    y las métricas (resultados/). Por defecto, la misma carpeta del dataset. Los
    equipos, que tienen el dataset compartido solo para leer, ponen aquí una
    carpeta suya con la variable de entorno P2M_SALIDAS."""
    return Path(os.environ.get("P2M_SALIDAS", carpeta_datos()))


def comprobar_salidas() -> Path:
    """La carpeta de salidas, revisando ANTES de trabajar que se puede escribir en ella.
    Sin P2M_SALIDAS es la del dataset, que a los equipos se les comparte solo para leer:
    así el error sale al inicio y no después de entrenar o de calcular vectores."""
    carpeta = carpeta_salidas()
    prueba = carpeta / ".p2m_prueba_escritura"
    try:
        carpeta.mkdir(parents=True, exist_ok=True)
        prueba.write_bytes(b"")
        prueba.unlink()
    except OSError as error:
        if "P2M_SALIDAS" in os.environ:
            consejo = "Revisa que la ruta de P2M_SALIDAS exista en tu Drive y sea tuya."
        else:
            consejo = ("No está definida P2M_SALIDAS, así que se intentó guardar en la carpeta del "
                       "dataset, que es solo de lectura. En la primera celda del notebook quita el # de "
                       "os.environ[\"P2M_SALIDAS\"] = \"/content/drive/MyDrive/poster2movie_equipoX\" "
                       "(con tu equipo) y vuelve a correr.")
        raise PermissionError(f"No puedo escribir en '{carpeta}' ({error}). {consejo}") from error
    return carpeta


def buscar(relativa: str) -> Path:
    """Un archivo calculado (modelo, vectores): primero en las salidas propias y,
    si no está, en la carpeta del dataset (los del sistema base)."""
    for carpeta in (carpeta_salidas(), carpeta_datos()):
        if (carpeta / relativa).exists():
            return carpeta / relativa
    return carpeta_salidas() / relativa

# Reparto por franquicia: todas las películas de una misma colección de TMDB
# (o, si no tiene, la película sola) caen en el mismo split. Se decide con un
# hash del grupo, no al azar: así agregar películas nuevas después no cambia
# el split de las que ya estaban.
PORCENTAJE_VAL = 10
PORCENTAJE_TEST = 10


def grupo(id_pelicula: int, coleccion: str) -> str:
    return f"coleccion_{coleccion}" if coleccion else f"pelicula_{id_pelicula}"


def split_de(grupo_pelicula: str) -> str:
    cubeta = int(hashlib.md5(grupo_pelicula.encode()).hexdigest(), 16) % 100
    if cubeta < PORCENTAJE_TEST:
        return "test"
    if cubeta < PORCENTAJE_TEST + PORCENTAJE_VAL:
        return "val"
    return "train"
