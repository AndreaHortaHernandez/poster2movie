"""Quita de la carpeta de un equipo las películas que no pasan la validación por
falta de datos (sin géneros o con sinopsis demasiado corta): borra su fila del
metadata.csv y su cartel.

Uso:
    python scripts/quitar_incompletas.py equipo1            # muestra qué quitaría
    python scripts/quitar_incompletas.py equipo1 --borrar   # las quita
"""
import argparse
import csv

from comun import (COLUMNAS, CONTRIBUCIONES, EQUIPOS, MIN_PALABRAS_SINOPSIS,
                   leer_metadata, parsear_generos)


def incompleta(fila):
    try:
        sin_generos = not parsear_generos(fila["genres"])
    except ValueError:
        sin_generos = True
    return sin_generos or len(fila["overview"].split()) < MIN_PALABRAS_SINOPSIS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("equipo", choices=EQUIPOS)
    p.add_argument("--borrar", action="store_true", help="sin esto solo muestra qué quitaría")
    args = p.parse_args()

    carpeta = CONTRIBUCIONES / args.equipo
    filas = leer_metadata(args.equipo)
    malas = [f for f in filas if incompleta(f)]
    print(f"{args.equipo}: {len(malas)} de {len(filas)} películas incompletas")
    for f in malas[:10]:
        print(f"  {f['id']}: {f['title']}")
    if len(malas) > 10:
        print(f"  ... y {len(malas) - 10} más")

    if not args.borrar:
        print("\nNo se borró nada. Para quitarlas: agrega --borrar")
        return

    for f in malas:
        (carpeta / "posters" / f"{f['id']}.jpg").unlink(missing_ok=True)
    with open(carpeta / "metadata.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(f for f in filas if not incompleta(f))
    print(f"Quitadas {len(malas)}. Quedan {len(filas) - len(malas)}. Corre scripts/validar.py {args.equipo}")


if __name__ == "__main__":
    main()
