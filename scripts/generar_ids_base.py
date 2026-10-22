"""Genera data/ids_base.txt con los ids del dataset base de Hugging Face.

Solo lee la columna `id` de los archivos parquet (no descarga los 43 GB de imágenes).
Normalmente NO hace falta correrlo: data/ids_base.txt ya viene en el repositorio.
"""
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from comun import IDS_BASE

REPO = "datasets/skvarre/movie_posters-100k"


def main():
    fs = HfFileSystem()
    archivos = sorted(fs.glob(f"{REPO}/**/*.parquet"))
    ids = []
    for ruta in archivos:
        # Bloques chicos: solo se piden por HTTP los bytes de la columna id.
        with fs.open(ruta, "rb", block_size=256 * 1024) as f:
            ids += pq.ParquetFile(f).read(columns=["id"]).column("id").to_pylist()
        print(f"{ruta}: {len(ids)} ids acumulados", flush=True)
    IDS_BASE.parent.mkdir(exist_ok=True)
    IDS_BASE.write_text("\n".join(map(str, sorted(set(ids)))) + "\n")
    print(f"Guardados {len(set(ids))} ids en {IDS_BASE}")


if __name__ == "__main__":
    main()
