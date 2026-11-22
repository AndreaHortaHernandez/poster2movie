"""Junta lo que ha subido cada equipo, para el reporte final (rol A).

Saca tres cosas y las deja en una carpeta (por defecto reporte_equipos/):

1. Git: qué ramas hay, de qué equipo son, si ya están en main, cuántos commits, quién los hizo,
   qué archivos tocaron y la lista de commits (cada intento suele ser un commit con su
   explicación). También copia los RESULTADOS_*.md que cada equipo tenga en su rama.
   -> ramas.csv, commits.csv, resultados_md/
2. Dataset: películas aportadas por cada equipo (en main, o en la rama que se indique con --ref):
   cuántas, géneros, idiomas, años, largo de sinopsis, franquicias. Con una gráfica.
   -> dataset_equipos.csv, generos_por_equipo.csv, generos_por_equipo.png
3. Métricas (opcional): todos los <nombre>_metricas.csv que encuentre en las carpetas de
   resultados que se le den (la del sistema base y las de cada equipo en Drive), en una tabla.
   -> metricas_todas.csv

Y un resumen.md que junta todo en tablas.

Uso (desde la raíz del repo; necesita `git fetch` reciente para ver las ramas de todos):
    git fetch --all --prune
    python scripts/reporte_equipos.py
    python scripts/reporte_equipos.py --ref origin/equipo3-datos          # dataset de una rama
    python scripts/reporte_equipos.py --resultados "/content/drive/MyDrive/poster2movie_datos" \\
        "/content/drive/MyDrive/poster2movie_equipo4"                     # + métricas de Drive

En Colab:
    !git clone -q https://github.com/AndreaHortaHernandez/poster2movie.git && cd poster2movie
    from google.colab import drive; drive.mount("/content/drive")
    !python scripts/reporte_equipos.py --resultados /content/drive/MyDrive/poster2movie_datos \\
        /content/drive/MyDrive/poster2movie_equipo*  --salida /content/drive/MyDrive/reporte_equipos

Para las métricas, cada equipo tiene que compartir (como lector) su carpeta de salidas de Drive;
agréguenla a Mi unidad como acceso directo.
"""
import argparse
import csv
import io
import json
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import EQUIPOS, RAIZ, parsear_generos  # noqa: E402

GENEROS_DESCARTADOS = {"Music", "TV Movie"}


def git(*args):
    r = subprocess.run(["git", *args], cwd=RAIZ, capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", errors="replace")


def equipo_de(rama):
    m = re.match(r"(equipo\d+)", rama)
    return m.group(1) if m else "otro"


# ------------------------------------------------------------------------------- 1. git
def resumen_git(principal, salida):
    ramas = [r.strip() for r in (git("branch", "-r", "--format=%(refname:short)") or "").splitlines()]
    ramas = [r for r in ramas if r and r != principal and not r.endswith("/HEAD") and "/" in r]
    filas, commits = [], []
    dir_md = salida / "resultados_md"
    for ref in sorted(ramas):
        rama = ref.split("/", 1)[1]
        log = git("log", "--no-merges", "--date=short", "--format=%h\t%an\t%ad\t%s", f"{principal}..{ref}") or ""
        lineas = [l.split("\t", 3) for l in log.splitlines() if l.count("\t") >= 3]
        integrada = git("merge-base", "--is-ancestor", ref, principal) is not None
        archivos = (git("diff", "--name-only", f"{principal}...{ref}") or "").splitlines()
        carpetas = Counter("/".join(a.split("/")[:2]) if a.startswith("contribuciones/") else a.split("/")[0]
                           for a in archivos)
        fechas = sorted(c[2] for c in lineas)
        filas.append({
            "rama": rama, "equipo": equipo_de(rama), "en_main": "sí" if integrada else "no",
            "commits_propios": len(lineas), "autores": ", ".join(sorted({c[1] for c in lineas})),
            "primer_commit": fechas[0] if fechas else "", "ultimo_commit": fechas[-1] if fechas else "",
            "archivos_cambiados": len(archivos),
            "carpetas": "; ".join(f"{k} ({v})" for k, v in carpetas.most_common(8)),
        })
        for h, autor, fecha, msg in lineas:
            commits.append({"rama": rama, "equipo": equipo_de(rama), "commit": h, "autor": autor,
                            "fecha": fecha, "mensaje": msg})
        for a in archivos:
            if re.fullmatch(r"RESULTADOS[^/]*\.md", a):
                texto = git("show", f"{ref}:{a}")
                if texto:
                    dir_md.mkdir(parents=True, exist_ok=True)
                    (dir_md / f"{rama}__{a}").write_text(texto, encoding="utf-8")
    escribir_csv(salida / "ramas.csv", filas)
    escribir_csv(salida / "commits.csv", commits)
    return filas, commits


# --------------------------------------------------------------------------- 2. dataset
def leer_metadata_ref(ref, equipo):
    texto = git("show", f"{ref}:contribuciones/{equipo}/metadata.csv")
    return list(csv.DictReader(io.StringIO(texto))) if texto else []


def resumen_dataset(ref, salida):
    filas, generos_primero, generos_todos = [], {}, {}
    for eq in EQUIPOS:
        pelis = leer_metadata_ref(ref, eq)
        gens = [[g["name"] for g in parsear_generos(p["genres"]) if g["name"] not in GENEROS_DESCARTADOS]
                for p in pelis]
        generos_primero[eq] = Counter(g[0] for g in gens if g)
        generos_todos[eq] = Counter(x for g in gens for x in g)
        anios = [int(p["release_date"][:4]) for p in pelis if p.get("release_date", "")[:4].isdigit()]
        palabras = [len(p["overview"].split()) for p in pelis if p.get("overview")]
        idiomas = Counter(p.get("original_language", "") for p in pelis)
        filas.append({
            "equipo": eq, "peliculas": len(pelis),
            "con_franquicia_%": round(100 * sum(bool(p.get("coleccion")) for p in pelis) / len(pelis), 1) if pelis else 0,
            "anio_min": min(anios, default=""), "anio_mediana": int(statistics.median(anios)) if anios else "",
            "anio_max": max(anios, default=""),
            "desde_2023_%": round(100 * sum(a >= 2023 for a in anios) / len(anios), 1) if anios else 0,
            "palabras_sinopsis_media": round(statistics.mean(palabras), 1) if palabras else "",
            "idiomas_top": ", ".join(f"{k} {v}" for k, v in idiomas.most_common(6)),
            "generos_por_pelicula": round(statistics.mean(len(g) for g in gens), 2) if gens else "",
        })
    escribir_csv(salida / "dataset_equipos.csv", filas)

    generos = sorted(set().union(*generos_todos.values()))
    tabla = [{"genero": g, **{f"{eq}_primero": generos_primero[eq][g] for eq in EQUIPOS},
              **{f"{eq}_etiquetas": generos_todos[eq][g] for eq in EQUIPOS}} for g in generos]
    escribir_csv(salida / "generos_por_equipo.csv", tabla)
    grafica_generos(generos, generos_primero, salida / "generos_por_equipo.png", ref)
    return filas


def grafica_generos(generos, conteos, ruta, ref):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    abajo = [0] * len(generos)
    for eq in EQUIPOS:
        v = [conteos[eq][g] for g in generos]
        ax.bar(generos, v, bottom=abajo, label=eq)
        abajo = [a + b for a, b in zip(abajo, v)]
    ax.set_ylabel("películas (por su primer género)")
    ax.set_title(f"Películas aportadas por equipo ({ref})")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- 3. métricas
def resumen_metricas(carpetas, salida):
    filas = []
    for c in carpetas:
        c = Path(c)
        if not c.exists():
            print(f"  [!] no existe {c}")
            continue
        for f in sorted(c.rglob("*_metricas.csv")):
            with open(f, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    filas.append({"carpeta": c.name, "modulo": f.parent.name,
                                  "archivo": f.name, **r})
    if filas:
        escribir_csv(salida / "metricas_todas.csv", filas)
    return filas


# ------------------------------------------------------------------------------ salida
def escribir_csv(ruta, filas):
    if not filas:
        return
    campos = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def tabla_md(filas, columnas):
    if not filas:
        return "_(nada)_\n"
    lineas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for f in filas:
        lineas.append("| " + " | ".join(str(f.get(c, "")).replace("|", "/") for c in columnas) + " |")
    return "\n".join(lineas) + "\n"


def redondear(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    return str(int(x)) if x.is_integer() else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--principal", default="origin/main", help="rama contra la que se compara")
    ap.add_argument("--ref", default="origin/main", help="de qué rama se leen las películas de los equipos")
    ap.add_argument("--resultados", nargs="*", default=[],
                    help="carpetas de salidas en Drive (la base y las de cada equipo)")
    ap.add_argument("--salida", default=RAIZ / "reporte_equipos", type=Path)
    args = ap.parse_args()

    if git("rev-parse", "--verify", args.principal) is None:
        sys.exit(f"No encuentro {args.principal}. Corran `git fetch --all --prune` dentro del repo.")
    if shutil.which("git") is None:
        sys.exit("Hace falta git.")
    args.salida.mkdir(parents=True, exist_ok=True)

    ramas, commits = resumen_git(args.principal, args.salida)
    dataset = resumen_dataset(args.ref, args.salida)
    metricas = resumen_metricas(args.resultados, args.salida)

    fecha = (git("log", "-1", "--date=short", "--format=%ad", args.principal) or "").strip()
    md = [f"# Lo que ha subido cada equipo\n",
          f"Generado con `scripts/reporte_equipos.py`. `{args.principal}` al {fecha}; dataset leído de `{args.ref}`.\n",
          "## Ramas\n",
          tabla_md(ramas, ["rama", "equipo", "en_main", "commits_propios", "autores", "primer_commit",
                           "ultimo_commit", "archivos_cambiados"]),
          "Lista completa de commits (cada intento con su explicación) en `commits.csv`. "
          "Los `RESULTADOS_*.md` de cada rama están en `resultados_md/`.\n",
          "## Películas aportadas por equipo\n",
          tabla_md(dataset, ["equipo", "peliculas", "anio_mediana", "desde_2023_%", "palabras_sinopsis_media",
                             "con_franquicia_%", "idiomas_top"]),
          f"Total: {sum(d['peliculas'] for d in dataset):,} películas. Por género en `generos_por_equipo.csv` "
          "y `generos_por_equipo.png`.\n"]
    md.append("## Métricas encontradas\n")
    if metricas:
        for mod in sorted({m["modulo"] for m in metricas}):
            sub = [m for m in metricas if m["modulo"] == mod]
            todas = dict.fromkeys(c for m in sub for c in m)
            cols = ["carpeta", "modelo"] + [c for c in todas if c not in
                                            ("carpeta", "modulo", "archivo", "modelo", "generador")][:8]
            md.append(f"### {mod}\n")
            md.append(tabla_md([{k: redondear(v) for k, v in m.items()} for m in sub], cols))
    else:
        md.append("_No se dieron carpetas de resultados (`--resultados`) o no tienen `*_metricas.csv`._\n")
    (args.salida / "resumen.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Listo en {args.salida}:")
    print(f"  {len(ramas)} ramas, {len(commits)} commits, {sum(d['peliculas'] for d in dataset):,} películas, "
          f"{len(metricas)} filas de métricas")
    for f in sorted(args.salida.rglob("*")):
        if f.is_file():
            print("   ", f.relative_to(args.salida))


if __name__ == "__main__":
    main()
