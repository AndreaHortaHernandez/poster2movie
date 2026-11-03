"""
Construye el dataset compacto de Poster2Movie (se corre UNA vez, en Colab).

Junta el dataset base (skvarre/movie_posters-100k, 43 GB) con las películas
de los equipos (contribuciones/) y deja en `destino/`:

    base-0000.parquet ... base-0087.parquet   carteles del dataset base, uno por archivo original
    equipos.parquet                           carteles de los equipos
    metadata.parquet                          todas las filas SIN imagen (para analizar rápido)
    resumen.json                              conteos por split, fuente y género

Qué hace con cada película:
1. Quita repetidas: el dataset base trae la misma película hasta 10 veces
   (95,300 filas, solo 73,429 películas distintas).
2. Géneros: quita Music y TV Movie y se queda con los 17 de config.GENEROS.
   Si una película se queda sin géneros, se descarta.
3. Descarta las que tienen sinopsis de menos de 20 palabras.
4. Reduce el cartel a 384 px de ancho (JPEG). Si la imagen está rota, se descarta.
5. Le asigna train / val / test por franquicia (config.split_de).

Uso (Colab):
    python -m p2m.construir_dataset /content/drive/MyDrive/poster2movie_datos

Procesa un archivo del dataset base a la vez y lo borra al terminar (nunca
ocupa más de ~1 GB de disco). Si se interrumpe, volver a correrlo continúa
desde el último archivo terminado.
"""

import argparse
import csv
import io
import json
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from p2m import config

ESQUEMA = pa.schema([
    ("id", pa.int64()),
    ("title", pa.string()),
    ("overview", pa.string()),
    ("tagline", pa.string()),
    ("generos", pa.list_(pa.string())),
    ("anio", pa.int32()),
    ("idioma", pa.string()),
    ("popularidad", pa.float64()),
    ("coleccion", pa.string()),
    ("split", pa.string()),
    ("fuente", pa.string()),  # "base" o "equipoN"
    ("imagen", pa.binary()),  # JPEG de 384 px de ancho
])


# =========================================================
# LIMPIEZA DE UNA PELÍCULA
# =========================================================

def limpiar_generos(nombres: list[str]) -> list[str]:
    """Quita Music / TV Movie y deja los demás en el orden fijo de config.GENEROS."""
    presentes = set(nombres) - config.GENEROS_DESCARTADOS
    return [g for g in config.GENEROS if g in presentes]


def reducir_cartel(datos: bytes) -> bytes | None:
    """Cartel a ANCHO_CARTEL px de ancho manteniendo la proporción; None si está roto."""
    try:
        with Image.open(io.BytesIO(datos)) as imagen:
            imagen = imagen.convert("RGB")
            if imagen.width > config.ANCHO_CARTEL:
                alto = round(imagen.height * config.ANCHO_CARTEL / imagen.width)
                imagen = imagen.resize((config.ANCHO_CARTEL, alto), Image.LANCZOS)
            salida = io.BytesIO()
            imagen.save(salida, "JPEG", quality=config.CALIDAD_JPEG, optimize=True)
            return salida.getvalue()
    except Exception:
        return None


def armar_fila(datos: dict, colecciones: dict[int, str], fuente: str) -> dict | None:
    """`datos` trae id, title, overview, tagline, nombres de géneros, release_date,
    original_language, popularity e imagen (bytes originales). None = se descarta."""
    generos = limpiar_generos(datos["generos"])
    overview = (datos["overview"] or "").strip()
    if not generos or len(overview.split()) < config.MIN_PALABRAS_SINOPSIS:
        return None
    imagen = reducir_cartel(datos["imagen"]) if datos["imagen"] else None
    if imagen is None:
        return None
    coleccion = colecciones.get(datos["id"], "")
    fecha = datos["release_date"] or ""
    return {
        "id": datos["id"],
        "title": datos["title"] or "",
        "overview": overview,
        "tagline": datos["tagline"] or "",
        "generos": generos,
        "anio": int(fecha[:4]) if fecha[:4].isdigit() else None,
        "idioma": datos["original_language"] or "",
        "popularidad": float(datos["popularity"] or 0),
        "coleccion": coleccion,
        "split": config.split_de(config.grupo(datos["id"], coleccion)),
        "fuente": fuente,
        "imagen": imagen,
    }


def _escribir(filas: list[dict], ruta: Path) -> None:
    temporal = ruta.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(filas, schema=ESQUEMA), temporal)
    temporal.replace(ruta)  # así un archivo a medias nunca parece terminado


# =========================================================
# FUENTES
# =========================================================

def leer_colecciones() -> dict[int, str]:
    """id -> id de la colección de TMDB ("" si no tiene). Del dataset base sale de
    data/colecciones_base.csv; de los equipos, de su propio metadata.csv."""
    colecciones = {}
    if config.RUTA_COLECCIONES_BASE.exists():
        with open(config.RUTA_COLECCIONES_BASE, newline="", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                if fila["coleccion"].isdigit():
                    colecciones[int(fila["id"])] = fila["coleccion"]
    else:
        print("AVISO: falta data/colecciones_base.csv (scripts/obtener_colecciones.py). "
              "Las películas del dataset base se repartirán sin agrupar franquicias.")
    for metadata in (config.RAIZ / "contribuciones").glob("equipo*/metadata.csv"):
        with open(metadata, newline="", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                if fila.get("coleccion", "").isdigit():
                    colecciones[int(fila["id"])] = fila["coleccion"]
    return colecciones


def procesar_base(destino: Path, colecciones: dict[int, str], hilos: int) -> set[int]:
    from huggingface_hub import HfApi, hf_hub_download

    archivos = sorted(
        f for f in HfApi().list_repo_files(config.DATASET_BASE, repo_type="dataset")
        if f.endswith(".parquet")
    )
    vistos: set[int] = set()
    for terminado in sorted(destino.glob("base-*.parquet")):
        vistos |= set(pq.read_table(terminado, columns=["id"]).column("id").to_pylist())

    with tempfile.TemporaryDirectory() as temporal:
        for numero, archivo in enumerate(archivos):
            salida = destino / f"base-{numero:04d}.parquet"
            if salida.exists():
                continue
            ruta = Path(hf_hub_download(config.DATASET_BASE, archivo, repo_type="dataset", local_dir=temporal))
            tabla = pq.read_table(ruta)
            ruta.unlink()  # el original (~500 MB) ya no hace falta

            candidatas = []
            for fila in tabla.to_pylist():
                if fila["id"] in vistos:
                    continue  # repetida
                vistos.add(fila["id"])
                candidatas.append({
                    **fila,
                    "generos": [g["name"] for g in fila["genres"] or []],
                    "imagen": (fila["image"] or {}).get("bytes"),
                })
            with ThreadPoolExecutor(hilos) as pool:
                filas = [f for f in pool.map(lambda d: armar_fila(d, colecciones, "base"), candidatas) if f]
            _escribir(filas, salida)
            print(f"base {numero + 1}/{len(archivos)}: {tabla.num_rows} filas -> "
                  f"{len(filas)} películas útiles", flush=True)
    return vistos


def procesar_equipos(destino: Path, colecciones: dict[int, str], ids_base: set[int]) -> None:
    filas = []
    for metadata in sorted((config.RAIZ / "contribuciones").glob("equipo*/metadata.csv")):
        equipo = metadata.parent.name
        with open(metadata, newline="", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                id_pelicula = int(fila["id"])
                if id_pelicula in ids_base:
                    continue
                cartel = metadata.parent / "posters" / f"{id_pelicula}.jpg"
                try:
                    generos = [g["name"] for g in json.loads(fila["genres"] or "[]")]
                except ValueError:
                    generos = []
                pelicula = armar_fila({
                    "id": id_pelicula, "title": fila["title"], "overview": fila["overview"],
                    "tagline": fila["tagline"], "generos": generos,
                    "release_date": fila["release_date"], "original_language": fila["original_language"],
                    "popularity": fila["popularity"] or 0,
                    "imagen": cartel.read_bytes() if cartel.exists() else None,
                }, colecciones, equipo)
                if pelicula:
                    filas.append(pelicula)
    _escribir(filas, destino / "equipos.parquet")
    print(f"equipos: {len(filas)} películas útiles")


# =========================================================
# RESUMEN
# =========================================================

def resumir(destino: Path) -> dict:
    columnas = [c for c in ESQUEMA.names if c != "imagen"]
    partes = sorted(destino.glob("base-*.parquet")) + [destino / "equipos.parquet"]
    metadata = pa.concat_tables(pq.read_table(p, columns=columnas) for p in partes)
    pq.write_table(metadata, destino / "metadata.parquet")

    filas = metadata.to_pylist()
    resumen = {
        "peliculas": len(filas),
        "por_split": dict(Counter(f["split"] for f in filas)),
        "por_fuente": dict(Counter(f["fuente"] for f in filas)),
        "con_coleccion": sum(bool(f["coleccion"]) for f in filas),
        "por_genero": {
            split: dict(Counter(g for f in filas if f["split"] == split for g in f["generos"]))
            for split in ("train", "val", "test")
        },
    }
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    return resumen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("destino", type=Path, help="carpeta donde queda el dataset compacto")
    parser.add_argument("--hilos", type=int, default=8, help="carteles que se reducen a la vez")
    args = parser.parse_args()
    args.destino.mkdir(parents=True, exist_ok=True)

    colecciones = leer_colecciones()
    ids_base = procesar_base(args.destino, colecciones, args.hilos)
    procesar_equipos(args.destino, colecciones, ids_base)
    resumen = resumir(args.destino)
    print(json.dumps({k: v for k, v in resumen.items() if k != "por_genero"}, indent=1))


if __name__ == "__main__":
    main()
