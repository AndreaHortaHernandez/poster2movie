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

Si el .pt trae un archivo de vectores (`embeddings`) que existe, se evalúa con esos vectores
(segundos). Si no (modelos del repo del clasificador con varios codificadores, o vectores
que no están en esta máquina), se codifica cada cartel de test con el Clasificador
completo: mismas películas y mismas métricas, pero tarda minutos (en GPU).

Guarda en <carpeta de salidas>/resultados/clasificador/<nombre>_metricas.csv y <nombre>_por_genero.csv.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support

from p2m import config, datos
from p2m.clasificador import (LOTE, MODELO_BASE, NOMBRE_EMBEDDINGS, Clasificador, aplicar_umbrales,
                              cargar_cabeza, cargar_split, ruta_embeddings)

RESULTADOS = config.carpeta_salidas() / "resultados" / "clasificador"


def probabilidades_con_vectores(guardado: dict, embeddings: str) -> tuple[np.ndarray, np.ndarray]:
    """Rápido: la cabeza sobre los vectores ya calculados de test (el camino de siempre)."""
    x, y, _ = cargar_split("test", embeddings)
    capa = cargar_cabeza(guardado, x.shape[1])
    with torch.no_grad():
        return torch.sigmoid(capa(torch.tensor(x))).numpy(), y


def probabilidades_codificando(ruta_modelo) -> tuple[np.ndarray, np.ndarray]:
    """Lento: cada cartel de test pasa por el Clasificador completo (codificador + cabeza).
    Para los .pt que no tienen un solo archivo de vectores (varios codificadores,
    `embeddings = None`) o cuyos vectores no están en esta máquina."""
    clasificador = Clasificador(ruta_modelo)
    generos = datos.metadata("test").set_index("id")["generos"]
    prob, y, lote = [], [], []

    def procesar():
        prob.append(clasificador.probabilidades([datos.abrir_cartel(f["imagen"]) for f in lote]))
        y.extend(datos.vector_generos(list(generos[f["id"]])) for f in lote)
        lote.clear()

    for fila in datos.peliculas("test", columnas=["id", "imagen"]):
        lote.append(fila)
        if len(lote) == LOTE:
            procesar()
            if len(y) % (LOTE * 20) == 0:
                print(f"{len(y)} carteles de test codificados", flush=True)
    if lote:
        procesar()
    return np.concatenate(prob), np.array(y)


def metricas(y: np.ndarray, prediccion: np.ndarray, probabilidades: np.ndarray | None) -> dict:
    return {
        "F1_macro": f1_score(y, prediccion, average="macro", zero_division=0),
        "F1_micro": f1_score(y, prediccion, average="micro", zero_division=0),
        "F1_por_pelicula": f1_score(y, prediccion, average="samples", zero_division=0),
        "mAP": average_precision_score(y, probabilidades, average="macro") if probabilidades is not None else np.nan,
        "acierto_exacto": accuracy_score(y, prediccion),
    }


def evaluar(ruta_modelo=None, nombre="base") -> pd.DataFrame:
    config.comprobar_salidas()
    ruta_modelo = ruta_modelo or config.buscar(MODELO_BASE)
    guardado = torch.load(ruta_modelo, map_location="cpu", weights_only=False)
    # El modelo dice con qué vectores se entrenó. None = varios codificadores (repo del clasificador).
    # Un combinado (ensamble/apilamiento, varias cabezas) siempre pasa por el Clasificador completo.
    embeddings = None if guardado.get("tipo") == "combinado" else guardado.get("embeddings", NOMBRE_EMBEDDINGS)
    if embeddings and ruta_embeddings(embeddings).exists():
        prob, y = probabilidades_con_vectores(guardado, embeddings)
        _, y_train, _ = cargar_split("train", embeddings)
    else:
        print(f"Sin vectores precalculados ({embeddings}): se codifican los carteles de test (más lento).")
        prob, y = probabilidades_codificando(ruta_modelo)
        y_train = np.array([datos.vector_generos(list(g)) for g in datos.metadata("train")["generos"]])
    prediccion = aplicar_umbrales(prob, np.asarray(guardado["umbrales"]))

    # Línea base trivial: a todos los carteles, los 2 géneros más comunes en train.
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
    parser.add_argument("--modelo", default=None, help="por defecto, el clasificador base")
    parser.add_argument("--nombre", default="base")
    args = parser.parse_args()
    evaluar(args.modelo, args.nombre)
