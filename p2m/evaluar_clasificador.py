"""
Evaluación del clasificador en el split de TEST (siempre el mismo para todos).

Métricas (multi-etiqueta, 17 géneros):
- F1 macro: promedio del F1 de cada género sin pesar por frecuencia. Es la
  principal: si el modelo ignora Western o Guerra, baja.
- F1 micro: cuenta todos los aciertos juntos; la dominan Drama y Comedia.
- F1 por película (samples): qué tan bien acierta el conjunto de géneros de cada cartel.
- mAP: precisión promedio por género, sin depender de los umbrales.
- Acierto exacto: % de carteles con TODOS sus géneros correctos (muy estricto).

Además mide una línea base trivial (predecir siempre los géneros más comunes)
para que se vea cuánto aporta el modelo.

Uso:
    python -m p2m.evaluar_clasificador                       # el modelo base
    python -m p2m.evaluar_clasificador --modelo mi.pt --nombre equipoX_v2

Guarda en <carpeta del dataset>/resultados/clasificador/<nombre>_metricas.csv y <nombre>_por_genero.csv.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support

from p2m import config
from p2m.clasificador import RUTA_MODELO, aplicar_umbrales, cargar_split

RESULTADOS = config.carpeta_datos() / "resultados" / "clasificador"


def metricas(y: np.ndarray, prediccion: np.ndarray, probabilidades: np.ndarray | None) -> dict:
    return {
        "F1_macro": f1_score(y, prediccion, average="macro", zero_division=0),
        "F1_micro": f1_score(y, prediccion, average="micro", zero_division=0),
        "F1_por_pelicula": f1_score(y, prediccion, average="samples", zero_division=0),
        "mAP": average_precision_score(y, probabilidades, average="macro") if probabilidades is not None else np.nan,
        "acierto_exacto": accuracy_score(y, prediccion),
    }


def evaluar(ruta_modelo=RUTA_MODELO, nombre="base") -> pd.DataFrame:
    x, y, _ = cargar_split("test")
    guardado = torch.load(ruta_modelo, map_location="cpu", weights_only=False)
    capa = torch.nn.Linear(x.shape[1], len(config.GENEROS))
    capa.load_state_dict(guardado["capa"])
    with torch.no_grad():
        prob = torch.sigmoid(capa(torch.tensor(x))).numpy()
    prediccion = aplicar_umbrales(prob, np.asarray(guardado["umbrales"]))

    # Línea base trivial: a todos los carteles, los 2 géneros más comunes en train.
    _, y_train, _ = cargar_split("train")
    comunes = np.argsort(-y_train.sum(axis=0))[:2]
    trivial = np.zeros_like(y)
    trivial[:, comunes] = 1

    tabla = pd.DataFrame([
        {"modelo": nombre, "peliculas_test": len(y), **metricas(y, prediccion, prob)},
        {"modelo": "trivial (siempre " + " + ".join(config.GENEROS[i] for i in comunes) + ")",
         "peliculas_test": len(y), **metricas(y, trivial, None)},
    ])
    precision, recall, f1, soporte = precision_recall_fscore_support(y, prediccion, zero_division=0)
    por_genero = pd.DataFrame({
        "genero": [config.GENEROS_ES[g] for g in config.GENEROS],
        "peliculas_test": soporte, "precision": precision, "recall": recall, "F1": f1,
        "umbral": guardado["umbrales"],
    }).sort_values("F1", ascending=False)

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RESULTADOS / f"{nombre}_metricas.csv", index=False)
    por_genero.to_csv(RESULTADOS / f"{nombre}_por_genero.csv", index=False)
    print(tabla.to_string(index=False, float_format="{:.3f}".format))
    print("\n" + por_genero.to_string(index=False, float_format="{:.3f}".format))
    return tabla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default=RUTA_MODELO)
    parser.add_argument("--nombre", default="base")
    args = parser.parse_args()
    evaluar(args.modelo, args.nombre)
