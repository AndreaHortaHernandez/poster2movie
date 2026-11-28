"""
Evaluación del cartel en VAL (interina, hasta que `evaluar_cartel` tenga --split val).

Hace lo mismo que `python -m p2m.evaluar_cartel` (mismo juez, mismas métricas), pero la muestra sale
del split de VALIDACIÓN, con el mismo criterio que la de test: solo películas del dataset base,
ordenadas por un hash del id (así los primeros N son siempre los mismos y las muestras son anidadas).
Además mide lo que pide el protocolo de experimentos: segundos por cartel y memoria pico de GPU.

Desde la línea de comandos:
    python -m p2m.evaluar_cartel_val --cuantos 100 --nombre val_sdxl_base
    python -m p2m.evaluar_cartel_val --cuantos 100 --nombre val_otro --modelo stabilityai/sd-turbo

Desde Python (para experimentos que cambian el prompt, el tamaño o los pasos: se cambian los valores
de `p2m.cartel` antes de llamar, igual que en el evaluador normal):
    from p2m import cartel, evaluar_cartel_val
    cartel.ALTO = 768
    evaluar_cartel_val.evaluar_val(nombre="val_768", cuantos=100)

Los archivos quedan en <salidas>/resultados/cartel/<nombre>_metricas.csv, _ejemplos.csv, _lamina.jpg
y <nombre>_extra.csv (segundos por cartel y memoria pico).
"""

import argparse
import hashlib
import time

import numpy as np
import pandas as pd
import torch

from p2m import datos
from p2m import evaluar_cartel as ev
from p2m.evaluar_resena import SEMILLA


def muestra_val(cuantos: int) -> list[dict]:
    """Igual que muestra_test, pero del split de validación."""
    meta = datos.metadata("val")
    candidatas = meta.loc[meta["fuente"] == "base", "id"].tolist()
    candidatas.sort(key=lambda i: hashlib.md5(f"{SEMILLA}-{i}".encode()).hexdigest())
    ids = set(candidatas[:cuantos])
    columnas = ["id", "title", "generos", "overview", "imagen"]
    return sorted((f for f in datos.peliculas("val", columnas) if f["id"] in ids), key=lambda f: f["id"])


def f1_por_pelicula(esperados, detectados) -> np.ndarray:
    """F1 de cada película (su media es el F1 "samples" del evaluador): permite intervalos de confianza."""
    esperados, detectados = np.asarray(esperados), np.asarray(detectados)
    aciertos = (esperados * detectados).sum(axis=1)
    total = esperados.sum(axis=1) + detectados.sum(axis=1)
    return np.where(total > 0, 2 * aciertos / np.maximum(total, 1), 0.0)


def evaluar_val(modelo=ev.MODELO_CARTEL, nombre="val_base", cuantos=100, juez=None, split="val") -> pd.DataFrame:
    """split="val": muestra de validación (experimentos). split="test": la muestra OFICIAL de test del
    evaluador (la versión final se mide UNA sola vez ahí, con --cuantos 500, base incluido)."""
    assert split in ("val", "test"), split
    tiempo = {"segundos": 0.0, "n": 0}
    generar_original = ev.Cartelista.generar

    def generar_medido(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inicio = time.time()
        imagen = generar_original(self, *args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tiempo["segundos"] += time.time() - inicio
        tiempo["n"] += 1
        return imagen

    # el evaluador no guarda el F1 por película: se anotan los argumentos de sus dos llamadas a f1_score
    llamadas, f1_original = [], ev.f1_score

    def f1_registrado(y, p, **kwargs):
        llamadas.append((np.asarray(y), np.asarray(p)))
        return f1_original(y, p, **kwargs)

    muestra_original = ev.muestra_test
    if split == "val":
        ev.muestra_test = muestra_val      # el evaluador normal llama a muestra_test(cuantos)
    ev.f1_score = f1_registrado
    ev.Cartelista.generar = generar_medido
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        tabla = ev.evaluar(modelo, nombre, cuantos, juez)
    finally:
        ev.muestra_test = muestra_original
        ev.f1_score = f1_original
        ev.Cartelista.generar = generar_original

    if len(llamadas) >= 2:  # 1.ª llamada: cartel generado; 2.ª: cartel real. Se agregan al CSV de ejemplos
        ruta = ev.RESULTADOS / f"{nombre}_ejemplos.csv"
        ejemplos = pd.read_csv(ruta)
        ejemplos["F1_generado"] = f1_por_pelicula(*llamadas[0])
        ejemplos["F1_real"] = f1_por_pelicula(*llamadas[1])
        ejemplos.to_csv(ruta, index=False)

    pico = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    extra = pd.DataFrame([{
        "modelo": nombre, "split": split, "peliculas": tiempo["n"],
        "segundos_por_cartel": tiempo["segundos"] / max(tiempo["n"], 1),
        "memoria_pico_GB": pico,
    }])
    extra.to_csv(ev.RESULTADOS / f"{nombre}_extra.csv", index=False)
    print(extra.to_string(index=False, float_format="{:.2f}".format))
    return tabla


def evaluar_test(modelo=ev.MODELO_CARTEL, nombre="test_base", cuantos=500, juez=None) -> pd.DataFrame:
    """Medición en TEST (muestra oficial del evaluador), con F1 por película, segundos y memoria pico."""
    return evaluar_val(modelo, nombre, cuantos, juez, split="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default=ev.MODELO_CARTEL)
    parser.add_argument("--nombre", default="val_base")
    parser.add_argument("--cuantos", type=int, default=100)
    parser.add_argument("--juez", default=None)
    args = parser.parse_args()
    evaluar_val(args.modelo, args.nombre, args.cuantos, args.juez)
