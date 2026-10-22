"""Revisa las contribuciones de todos los equipos antes de subirlas.

Uso:
    python scripts/validar.py              # revisa los 4 equipos
    python scripts/validar.py equipo2      # solo un equipo

Termina con código 1 si encuentra errores (no subas hasta que diga OK).
"""
import sys

from PIL import Image

from comun import (COLUMNAS, CONTRIBUCIONES, EQUIPOS, MIN_PALABRAS_SINOPSIS,
                   ids_base, leer_metadata, parsear_generos)


def validar_equipo(equipo, base, vistos):
    errores = []
    carpeta = CONTRIBUCIONES / equipo
    filas = leer_metadata(equipo)
    if not filas:
        return errores, 0
    if list(filas[0].keys()) != COLUMNAS:
        errores.append(f"columnas incorrectas: {list(filas[0].keys())} (esperadas {COLUMNAS})")
        if "coleccion" not in filas[0]:
            errores.append("falta la columna 'coleccion': corre descargar_tmdb.py otra vez "
                           "(con --max 0) y la agrega sola")
        return errores, len(filas)

    ids_csv = set()
    for n, r in enumerate(filas, start=2):
        pre = f"fila {n} (id {r['id']})"
        try:
            i = int(r["id"])
        except ValueError:
            errores.append(f"{pre}: id no es número")
            continue
        if i in ids_csv:
            errores.append(f"{pre}: id repetido dentro del CSV")
        ids_csv.add(i)
        if i in base:
            errores.append(f"{pre}: ya está en el dataset base")
        if i in vistos and vistos[i] != equipo:
            errores.append(f"{pre}: también está en {vistos[i]}")
        vistos[i] = equipo
        if r["equipo"] != equipo:
            errores.append(f"{pre}: columna equipo='{r['equipo']}'")
        try:
            if not parsear_generos(r["genres"]):
                errores.append(f"{pre}: sin géneros")
        except ValueError:
            errores.append(f"{pre}: genres no es JSON válido")
        if r["coleccion"] and not r["coleccion"].isdigit():
            errores.append(f"{pre}: coleccion no es un id numérico")
        if len(r["overview"].split()) < MIN_PALABRAS_SINOPSIS:
            errores.append(f"{pre}: sinopsis demasiado corta")
        img = carpeta / "posters" / f"{i}.jpg"
        if not img.exists():
            errores.append(f"{pre}: falta posters/{i}.jpg")
        else:
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception:
                errores.append(f"{pre}: imagen corrupta")

    sobrantes = {p.stem for p in (carpeta / "posters").glob("*.jpg")} - {str(i) for i in ids_csv}
    for s in sorted(sobrantes):
        errores.append(f"posters/{s}.jpg no aparece en metadata.csv")
    return errores, len(filas)


def main():
    equipos = sys.argv[1:] or EQUIPOS
    base = ids_base()
    if not base:
        print("AVISO: falta data/ids_base.txt, no se revisan duplicados con el dataset base.\n")
    vistos = {}
    # Registrar ids de todos los equipos para detectar cruces aunque se valide uno solo.
    for eq in EQUIPOS:
        for r in leer_metadata(eq):
            vistos.setdefault(int(r["id"]), eq)

    total_errores = 0
    for eq in equipos:
        errores, n = validar_equipo(eq, base, dict(vistos))
        estado = "OK" if not errores else f"{len(errores)} errores"
        print(f"{eq}: {n} películas — {estado}")
        for e in errores[:30]:
            print("   -", e)
        if len(errores) > 30:
            print(f"   ... y {len(errores) - 30} más")
        total_errores += len(errores)
    sys.exit(1 if total_errores else 0)


if __name__ == "__main__":
    main()
