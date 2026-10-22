"""Constantes y utilidades compartidas por los scripts del dataset."""
import csv
import json
import sys
from pathlib import Path

# Evita que títulos con acentos/kanji truenen la consola de Windows.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ = Path(__file__).resolve().parent.parent
CONTRIBUCIONES = RAIZ / "contribuciones"
IDS_BASE = RAIZ / "data" / "ids_base.txt"

EQUIPOS = ["equipo1", "equipo2", "equipo3", "equipo4"]

# Mismas columnas que skvarre/movie_posters-100k (sin la imagen) + coleccion + equipo.
# `genres` se guarda como JSON: [{"id": 28, "name": "Action"}, ...]
# `coleccion` es el id de TMDB de la franquicia (belongs_to_collection); vacío si no tiene.
COLUMNAS = [
    "id", "title", "genres", "overview", "popularity", "release_date",
    "budget", "revenue", "tagline", "original_language", "runtime", "coleccion", "equipo",
]

# Géneros de TMDB (id -> nombre), los mismos que usa el dataset base.
GENEROS_TMDB = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}

MIN_PALABRAS_SINOPSIS = 20


def leer_metadata(equipo):
    ruta = CONTRIBUCIONES / equipo / "metadata.csv"
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ids_base():
    if not IDS_BASE.exists():
        return set()
    return {int(l) for l in IDS_BASE.read_text().split() if l.strip()}


def ids_contribuciones(excepto=None):
    ids = set()
    for eq in EQUIPOS:
        if eq != excepto:
            ids |= {int(r["id"]) for r in leer_metadata(eq)}
    return ids


def parsear_generos(texto):
    return json.loads(texto) if texto else []
