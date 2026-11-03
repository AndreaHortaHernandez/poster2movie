"""Pide a TMDB la franquicia (belongs_to_collection) de cada película del dataset base.

El dataset base no trae esa columna y la necesitamos para separar train / val / test
por franquicia (que "Toy Story" y "Toy Story 2" no queden una en train y otra en test).
Se corre UNA sola vez; el resultado (data/colecciones_base.csv) se sube al repo.

Uso:
    python scripts/obtener_colecciones.py

~73 mil consultas (~30-40 min). Se puede interrumpir con Ctrl+C y volver a correr:
continúa donde se quedó.
"""
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from comun import RAIZ, ids_base
from descargar_tmdb import coleccion, get, sesion_tmdb

SALIDA = RAIZ / "data" / "colecciones_base.csv"
HILOS = 8  # TMDB permite ~50 consultas por segundo
_local = threading.local()


def consultar(id_pelicula):
    if not hasattr(_local, "sesion"):
        _local.sesion = sesion_tmdb()
    try:
        return id_pelicula, coleccion(get(_local.sesion, f"/movie/{id_pelicula}"))
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == 404:
            return id_pelicula, "no_existe"  # la quitaron de TMDB
        raise


def main():
    sesion_tmdb()  # sale con un mensaje claro si falta TMDB_API_KEY
    hechos = set()
    if SALIDA.exists():
        with open(SALIDA, newline="", encoding="utf-8") as f:
            hechos = {int(r["id"]) for r in csv.DictReader(f)}
    pendientes = sorted(ids_base() - hechos)
    print(f"{len(hechos)} ya consultadas, faltan {len(pendientes)}")
    if not pendientes:
        return

    nuevo = not SALIDA.exists()
    with open(SALIDA, "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(HILOS) as hilos:
        escritor = csv.writer(f)
        if nuevo:
            escritor.writerow(["id", "coleccion"])
        inicio = time.time()
        for n, (id_pelicula, col) in enumerate(hilos.map(consultar, pendientes), start=1):
            escritor.writerow([id_pelicula, col])
            if n % 1000 == 0:
                f.flush()
                ritmo = n / (time.time() - inicio)
                print(f"{n}/{len(pendientes)} - faltan ~{(len(pendientes) - n) / ritmo / 60:.0f} min", flush=True)
    print(f"Listo: {SALIDA}")


if __name__ == "__main__":
    main()
