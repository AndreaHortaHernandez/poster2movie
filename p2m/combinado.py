"""
Modelos combinados del equipo del clasificador: ensamble y apilamiento (su "contrato 4").

Copia de la parte de INFERENCIA de `clasificador/combinado.py` del repo
11909/Poster2Movie-Clasificador (commit 9761cd8). Debe combinar EXACTAMENTE igual que allá:
si cambian cómo se combina, hay que traer el cambio aquí.

Un .pt combinado junta las cabezas de varios experimentos (los "miembros", cada uno con
sus `vectores`), promedia sus probabilidades con pesos y, si tiene apilamiento, las
corrige con una etapa 2. Al final se aplican sus propios umbrales:

    cartel ─┬─ codificadores ─ cabeza 1 ─ 17 prob ─┐
            └─ codificadores ─ cabeza 2 ─ 17 prob ─┴─ promedio con pesos ─ [etapa 2] ─ umbrales

    {"tipo": "combinado", "generos", "miembros": [{"nombre", "vectores", "arquitectura", "capa"}],
     "pesos", "apilamiento": None | {"arquitectura", "capa", "residual"}, "umbrales", "vectores"}

Un .pt de UN experimento (con `vectores`) se trata como un combinado de un miembro con peso 1.
"""

import numpy as np
import torch

TIPO = "combinado"


def es_combinado(guardado: dict) -> bool:
    return guardado.get("tipo") == TIPO


def como_combinado(guardado: dict) -> dict:
    """Un .pt de un experimento (con `vectores`) como combinado de un miembro (mismo resultado)."""
    if es_combinado(guardado):
        return guardado
    vectores = list(guardado["vectores"])
    return {
        "tipo": TIPO, "generos": guardado["generos"], "umbrales": guardado["umbrales"],
        "miembros": [{"nombre": "", "vectores": vectores,
                      "arquitectura": guardado.get("arquitectura") or {"tipo": "lineal"},
                      "capa": guardado["capa"]}],
        "pesos": [1.0], "apilamiento": None, "vectores": list(dict.fromkeys(vectores)),
    }


def cargar_cabezas(guardado: dict, dims: dict[str, int]) -> list[torch.nn.Module]:
    """Una cabeza por miembro. `dims`: dimensión de cada archivo de vectores."""
    from p2m.clasificador import construir_cabeza

    cabezas = []
    for m in guardado["miembros"]:
        cabeza = construir_cabeza(m["arquitectura"], sum(dims[v] for v in m["vectores"]), len(guardado["generos"]))
        cabeza.load_state_dict(m["capa"])
        cabezas.append(cabeza.eval())
    return cabezas


def cargar_etapa2(guardado: dict) -> torch.nn.Module | None:
    if not guardado.get("apilamiento"):
        return None
    from p2m.clasificador import construir_cabeza

    n = len(guardado["generos"])
    etapa2 = construir_cabeza(guardado["apilamiento"]["arquitectura"], n, n)
    etapa2.load_state_dict(guardado["apilamiento"]["capa"])
    return etapa2.eval()


@torch.no_grad()
def probabilidades(guardado: dict, cabezas: list, etapa2, vectores: dict[str, np.ndarray]) -> np.ndarray:
    """Probabilidades finales (n × 17) a partir de los vectores de cada archivo (n × D)."""
    promedio = 0.0
    for m, cabeza, peso in zip(guardado["miembros"], cabezas, guardado["pesos"]):
        x = torch.tensor(np.concatenate([vectores[v] for v in m["vectores"]], axis=1), dtype=torch.float32)
        promedio = promedio + peso * torch.sigmoid(cabeza(x))
    if etapa2 is None:
        return promedio.numpy()
    ajuste = etapa2(promedio)
    if guardado["apilamiento"].get("residual", True):
        ajuste = ajuste + torch.logit(promedio.clamp(1e-6, 1 - 1e-6))
    return torch.sigmoid(ajuste).numpy()
