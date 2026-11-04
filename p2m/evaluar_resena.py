"""
Evaluación del módulo de reseña en una muestra FIJA del split de test.

Para cada cartel se genera una sinopsis y se compara con la real (overview):
- ROUGE-L: qué tanto coinciden en palabras, en orden (0-1). Castiga los
  sinónimos aunque estén bien.
- BERTScore F1: similitud de significado con un Transformer (RoBERTa), aunque
  se usen otras palabras (reescalado: 0 ~ texto al azar, 1 ~ idéntico).
- Palabras: largo promedio (las sinopsis reales tienen ~40-70).

Por defecto se le dan los géneros REALES (así se mide solo la reseña, sin los
errores del clasificador); con --generos predichos se usan los del clasificador.

Ojo al interpretar: si el cartel trae el título y el modelo conoce la
película, puede "recordar" la sinopsis en vez de inferirla del cartel.

Uso:
    python -m p2m.evaluar_resena                         # 200 carteles, modelo base
    python -m p2m.evaluar_resena --cuantos 50 --nombre equipoX_lora --modelo ruta/o/id

Guarda en <carpeta de salidas>/resultados/resena/<nombre>_metricas.csv y <nombre>_ejemplos.csv.
"""

import argparse
import hashlib
import time

import pandas as pd

from p2m import config, datos
from p2m.resena import MODELO_RESENA, Resenador

RESULTADOS = config.carpeta_salidas() / "resultados" / "resena"
SEMILLA = 42


def muestra_test(cuantos: int) -> list[dict]:
    """Siempre las mismas películas de test (para comparar versiones entre sí).

    Solo se sortea entre las del dataset BASE (fuente == "base"), que no cambian cuando
    los equipos agregan películas, y se ordenan por un hash del id (no por un shuffle,
    que cambia si cambia el conjunto). Así reconstruir el dataset con películas nuevas
    no cambia la muestra."""
    meta = datos.metadata("test")
    candidatas = meta.loc[meta["fuente"] == "base", "id"].tolist()
    candidatas.sort(key=lambda i: hashlib.md5(f"{SEMILLA}-{i}".encode()).hexdigest())
    ids = set(candidatas[:cuantos])
    columnas = ["id", "title", "generos", "overview", "imagen"]
    return sorted((f for f in datos.peliculas("test", columnas) if f["id"] in ids), key=lambda f: f["id"])


def evaluar(modelo=MODELO_RESENA, nombre="base", cuantos=200, generos="reales") -> pd.DataFrame:
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer

    peliculas = muestra_test(cuantos)
    resenador = Resenador(modelo)
    clasificador = None
    if generos == "predichos":
        from p2m.clasificador import Clasificador
        clasificador = Clasificador()

    generadas, inicio = [], time.time()
    for n, f in enumerate(peliculas, start=1):
        cartel = datos.abrir_cartel(f["imagen"])
        pista = clasificador.predecir(cartel)["generos"] if clasificador else list(f["generos"])
        generadas.append(resenador.generar(cartel, pista))
        if n % 20 == 0:
            print(f"{n}/{len(peliculas)} ({time.time() - inicio:.0f}s)", flush=True)
    del resenador  # libera la GPU para BERTScore

    reales = [f["overview"] for f in peliculas]
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = [rouge.score(r, g)["rougeL"].fmeasure for r, g in zip(reales, generadas)]
    _, _, bert_f1 = bert_score(generadas, reales, lang="en", rescale_with_baseline=True, verbose=False)

    ejemplos = pd.DataFrame({
        "id": [f["id"] for f in peliculas], "titulo": [f["title"] for f in peliculas],
        "generos": [", ".join(f["generos"]) for f in peliculas],
        "generada": generadas, "real": reales, "ROUGE_L": rouge_l, "BERTScore": bert_f1.tolist(),
    })
    tabla = pd.DataFrame([{
        "modelo": nombre, "generador": str(modelo), "generos": generos, "carteles": len(peliculas),
        "ROUGE_L": ejemplos["ROUGE_L"].mean(), "BERTScore": ejemplos["BERTScore"].mean(),
        "palabras_generada": ejemplos["generada"].str.split().str.len().mean(),
        "palabras_real": ejemplos["real"].str.split().str.len().mean(),
    }])

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RESULTADOS / f"{nombre}_metricas.csv", index=False)
    ejemplos.to_csv(RESULTADOS / f"{nombre}_ejemplos.csv", index=False)
    print(tabla.to_string(index=False, float_format="{:.3f}".format))
    return tabla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default=MODELO_RESENA)
    parser.add_argument("--nombre", default="base")
    parser.add_argument("--cuantos", type=int, default=200)
    parser.add_argument("--generos", choices=["reales", "predichos"], default="reales")
    args = parser.parse_args()
    evaluar(args.modelo, args.nombre, args.cuantos, args.generos)
