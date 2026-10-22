"""Descarga películas de TMDB para la contribución de un equipo.

Uso:
    set TMDB_API_KEY=tu_api_key        (Windows)   /   export TMDB_API_KEY=...  (Mac/Linux)
    python scripts/descargar_tmdb.py --equipo equipo1 --generos 28 12 --max 300

Opciones útiles:
    --desde 2023-01-01        solo películas estrenadas desde esa fecha
    --idioma es               solo películas con idioma original español (cine mexicano/latino)
    --pais MX                 solo películas producidas en ese país

Reglas que aplica automáticamente:
  * La película se asigna por su PRIMER género: solo se guarda si ese primer
    género está en --generos (así dos equipos nunca agregan la misma película).
  * Se salta si el id ya está en el dataset base o en la carpeta de otro equipo.
  * Se descarta si no tiene cartel, no tiene géneros o la sinopsis es muy corta.
  * Se puede detener (Ctrl+C) y volver a correr: continúa donde se quedó.
"""
import argparse
import csv
import json
import os
import sys
import time

import requests

from comun import (COLUMNAS, CONTRIBUCIONES, GENEROS_TMDB, MIN_PALABRAS_SINOPSIS,
                   ids_base, ids_contribuciones, leer_metadata)

API = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p/w500"


def sesion_tmdb():
    clave = os.environ.get("TMDB_API_KEY")
    if not clave:
        sys.exit("Falta la variable de entorno TMDB_API_KEY (ver README).")
    s = requests.Session()
    if clave.startswith("eyJ"):  # token de lectura (v4)
        s.headers["Authorization"] = f"Bearer {clave}"
    else:  # api key (v3)
        s.params = {"api_key": clave}
    return s


def get(s, ruta, **params):
    for intento in range(5):
        r = s.get(API + ruta, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 * (intento + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def candidatos(s, args):
    params = {
        "with_genres": "|".join(map(str, args.generos)),
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "language": "en-US",
        "vote_count.gte": args.min_votos,
    }
    if args.desde:
        params["primary_release_date.gte"] = args.desde
    if args.hasta:
        params["primary_release_date.lte"] = args.hasta
    if args.idioma:
        params["with_original_language"] = args.idioma
    if args.pais:
        params["with_origin_country"] = args.pais
    pagina = 1
    while True:
        datos = get(s, "/discover/movie", page=pagina, **params)
        yield from datos["results"]
        if pagina >= min(datos["total_pages"], 500):
            return
        pagina += 1


def coleccion(d):
    return (d.get("belongs_to_collection") or {}).get("id", "")


def migrar_csv(s, ruta_csv):
    """Agrega la columna `coleccion` a un metadata.csv creado con la versión anterior."""
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas or "coleccion" in filas[0]:
        return
    print(f"Agregando columna 'coleccion' a {len(filas)} películas ya descargadas...")
    for r in filas:
        r["coleccion"] = coleccion(get(s, f"/movie/{r['id']}"))
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--equipo", required=True, choices=["equipo1", "equipo2", "equipo3", "equipo4"])
    p.add_argument("--generos", required=True, type=int, nargs="+",
                   help="ids de TMDB de los géneros del equipo: " +
                        ", ".join(f"{k}={v}" for k, v in GENEROS_TMDB.items()))
    p.add_argument("--max", type=int, default=200, help="cuántas películas NUEVAS agregar en esta corrida")
    p.add_argument("--desde")
    p.add_argument("--hasta")
    p.add_argument("--idioma")
    p.add_argument("--pais")
    p.add_argument("--min-votos", type=int, default=20)
    args = p.parse_args()

    for g in args.generos:
        if g not in GENEROS_TMDB:
            sys.exit(f"Género {g} no existe. Opciones: {GENEROS_TMDB}")

    carpeta = CONTRIBUCIONES / args.equipo
    posters = carpeta / "posters"
    posters.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta / "metadata.csv"

    s = sesion_tmdb()
    if ruta_csv.exists():
        migrar_csv(s, ruta_csv)

    existentes = {int(r["id"]) for r in leer_metadata(args.equipo)}
    excluir = ids_base() | ids_contribuciones(excepto=args.equipo) | existentes
    if not ids_base():
        print("AVISO: no se encontró data/ids_base.txt; no se podrán filtrar duplicados del dataset base.")

    nuevo = not ruta_csv.exists()
    agregadas = 0
    with open(ruta_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        for c in candidatos(s, args):
            if agregadas >= args.max:
                break
            if c["id"] in excluir or not c.get("poster_path"):
                continue
            excluir.add(c["id"])
            d = get(s, f"/movie/{c['id']}", language="en-US")
            generos = d.get("genres") or []
            if not generos or generos[0]["id"] not in args.generos:
                continue
            sinopsis = (d.get("overview") or "").strip()
            if len(sinopsis.split()) < MIN_PALABRAS_SINOPSIS:
                continue
            img = s.get(IMG + d["poster_path"], timeout=60)
            if img.status_code != 200:
                continue
            (posters / f"{d['id']}.jpg").write_bytes(img.content)
            w.writerow({
                "id": d["id"],
                "title": d.get("title", ""),
                "genres": json.dumps([{"id": g["id"], "name": g["name"]} for g in generos]),
                "overview": sinopsis,
                "popularity": d.get("popularity", 0),
                "release_date": d.get("release_date", ""),
                "budget": d.get("budget", 0),
                "revenue": d.get("revenue", 0),
                "tagline": d.get("tagline") or "",
                "original_language": d.get("original_language", ""),
                "runtime": d.get("runtime") or 0,
                "coleccion": coleccion(d),
                "equipo": args.equipo,
            })
            f.flush()
            agregadas += 1
            print(f"[{agregadas}/{args.max}] {d['id']} {d.get('title')} — {generos[0]['name']}")

    print(f"\nListo: {agregadas} películas nuevas en {carpeta}")
    print("Ahora corre:  python scripts/validar.py")


if __name__ == "__main__":
    main()
