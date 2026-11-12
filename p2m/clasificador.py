"""
Módulo 1 — Clasificador: cartel -> géneros (multi-etiqueta, 17 géneros).

Versión BASE (la que mejora el equipo del clasificador):

1. CLIP ViT-B/32 (Transformer de imágenes preentrenado por OpenAI con 400
   millones de pares imagen-texto) convierte cada cartel en un vector de 512
   números. Se queda congelado: no se entrena.
2. Una capa lineal 512 -> 17 da una probabilidad por género (sigmoide: cada
   género es un sí/no independiente, una película puede tener varios).
3. Un umbral por género, calibrado en validación para maximizar su F1: los
   géneros raros (Western 1 %) necesitan umbrales más bajos que Drama (44 %).

Como CLIP no se entrena, los vectores se calculan UNA vez para todo el
dataset (`calcular_embeddings`) y entrenar la capa lineal tarda segundos.

Uso:
    python -m p2m.clasificador embeddings   # una vez, con GPU (~20-30 min las 60 mil)
    python -m p2m.clasificador entrenar     # segundos
    python -m p2m.evaluar_clasificador      # métricas en test

Para los demás módulos, la interfaz es:
    clasificador = Clasificador()
    clasificador.predecir(imagen_pil)  ->  {"generos": ["Horror", ...], "probabilidades": {...}}
"""

import argparse
import hashlib
import os
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score
from transformers import CLIPModel, CLIPProcessor

from p2m import config, datos

MODELO_CLIP = "openai/clip-vit-base-patch32"
MODELO_BASE = "modelos/clasificador_base.pt"  # relativo a la carpeta de salidas / del dataset
NOMBRE_EMBEDDINGS = "embeddings_clip_b32.npz"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
LOTE = 64


def ruta_embeddings(nombre: str = NOMBRE_EMBEDDINGS):
    """Los vectores propios si ya se calcularon; si no, los del sistema base."""
    return config.buscar(nombre)


def ruta_modelo_base():
    """El clasificador del sistema base, siempre el de la carpeta del dataset (nunca uno
    propio de las salidas de un equipo). Es el "juez" que usa evaluar_cartel."""
    ruta = config.carpeta_datos() / MODELO_BASE
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro el clasificador base en '{ruta}'.")
    return ruta


# =========================================================
# MODELO DEL EQUIPO DEL CLASIFICADOR (releases de su repo)
# =========================================================
# El modelo que usa `Clasificador()` (la demo y evaluar_resena --generos predichos).
# Se descarga del release de GitHub de ese tag a una caché local, una sola vez.
# Para cambiar de modelo se cambia este valor en un PR. None = el clasificador base.
#   {"tag": "v0.2.0", "archivo": "<experimento>.pt", "sha256": "<opcional, se verifica>"}
REPO_CLASIFICADOR = "11909/Poster2Movie-Clasificador"
RELEASE_CLASIFICADOR: dict | None = None


def carpeta_cache() -> Path:
    """Dónde se guardan los .pt descargados: P2M_CACHE o ~/.cache/poster2movie."""
    return Path(os.environ.get("P2M_CACHE", Path.home() / ".cache" / "poster2movie"))


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def descargar_release(tag: str, archivo: str, sha256: str | None = None) -> Path:
    """El .pt `archivo` del release `tag` del repo del clasificador (de la caché si ya está)."""
    destino = carpeta_cache() / "clasificador" / tag / archivo
    if not destino.exists():
        url = f"https://github.com/{REPO_CLASIFICADOR}/releases/download/{tag}/{archivo}"
        print(f"Descargando el clasificador {tag} de {url}")
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_suffix(".descargando")
        urllib.request.urlretrieve(url, temporal)
        temporal.replace(destino)  # así una descarga cortada no queda como archivo bueno
    if sha256 and _sha256(destino) != sha256:
        destino.unlink()
        raise ValueError(f"El .pt descargado de {tag} no coincide con el sha256 esperado.")
    return destino


def ruta_modelo_clasificador() -> Path:
    """El .pt que carga `Clasificador()` por defecto: el release fijado en RELEASE_CLASIFICADOR.
    Si no hay release fijado, o no se puede descargar (sin red y sin caché), el base de siempre."""
    if RELEASE_CLASIFICADOR:
        try:
            return descargar_release(**RELEASE_CLASIFICADOR)
        except (OSError, ValueError) as error:
            warnings.warn(f"AVISO: no pude obtener el clasificador {RELEASE_CLASIFICADOR['tag']} ({error}). "
                          "Se usa el clasificador BASE.", stacklevel=2)
    return config.buscar(MODELO_BASE)


ARQUITECTURA_BASE = {"tipo": "lineal"}


def construir_cabeza(arquitectura: dict | None, entrada: int, salida: int) -> torch.nn.Module:
    """La parte entrenable del clasificador, según la arquitectura guardada en el .pt.

    {"tipo": "lineal"}                      capa lineal entrada -> salida (la base)
    {"tipo": "mlp", "ocultas": [512, 256]}  MLP con ReLU y dropout entre capas

    Un .pt sin arquitectura (los anteriores a este cambio) es lineal. Si un equipo
    necesita otra forma, agrega aquí su tipo: el evaluador y la demo lo reconstruyen solos."""
    arquitectura = arquitectura or ARQUITECTURA_BASE
    if arquitectura["tipo"] == "lineal":
        return torch.nn.Linear(entrada, salida)
    if arquitectura["tipo"] == "mlp":
        capas, anterior = [], entrada
        for oculta in arquitectura["ocultas"]:
            capas += [torch.nn.Linear(anterior, oculta), torch.nn.ReLU(),
                      torch.nn.Dropout(arquitectura.get("dropout", 0.2))]
            anterior = oculta
        return torch.nn.Sequential(*capas, torch.nn.Linear(anterior, salida))
    raise ValueError(f"Arquitectura desconocida: {arquitectura['tipo']}")


def cargar_cabeza(guardado: dict, entrada: int) -> torch.nn.Module:
    cabeza = construir_cabeza(guardado.get("arquitectura"), entrada, len(guardado["generos"]))
    cabeza.load_state_dict(guardado["capa"])
    return cabeza.eval()


class CodificadorCLIP:
    def __init__(self, modelo: str = MODELO_CLIP):
        self.procesador = CLIPProcessor.from_pretrained(modelo)
        self.modelo = CLIPModel.from_pretrained(modelo).to(DISPOSITIVO).eval()

    @torch.inference_mode()
    def codificar(self, imagenes) -> np.ndarray:
        entrada = self.procesador(images=imagenes, return_tensors="pt").to(DISPOSITIVO)
        vectores = self.modelo.get_image_features(**entrada)
        if not torch.is_tensor(vectores):  # transformers >= 5 regresa un objeto; el vector es pooler_output
            vectores = vectores.pooler_output
        return torch.nn.functional.normalize(vectores, dim=-1).float().cpu().numpy()

    @torch.inference_mode()
    def codificar_texto(self, textos: list[str]) -> np.ndarray:
        """Vector CLIP de textos (CLIP solo lee los primeros 77 tokens)."""
        entrada = self.procesador(text=textos, return_tensors="pt", padding=True,
                                  truncation=True, max_length=77).to(DISPOSITIVO)
        vectores = self.modelo.get_text_features(**entrada)
        if not torch.is_tensor(vectores):
            vectores = vectores.pooler_output
        return torch.nn.functional.normalize(vectores, dim=-1).float().cpu().numpy()


def calcular_embeddings() -> None:
    """Vector CLIP de cada cartel del dataset, guardado junto al dataset.

    Se guarda un pedazo por cada archivo del dataset (carpeta embeddings_partes/):
    si Colab se desconecta, al volver a correrlo continúa donde se quedó. En GPU
    tarda ~20-30 min; en CPU (si no hay cuota de GPU), ~1.5-2 h."""
    destino = config.comprobar_salidas() / NOMBRE_EMBEDDINGS
    codificador = CodificadorCLIP()
    partes = destino.parent / "embeddings_partes"
    partes.mkdir(parents=True, exist_ok=True)
    archivos = datos.archivos()
    for numero, archivo in enumerate(archivos, start=1):
        salida = partes / f"{archivo.stem}.npz"
        # Se salta si ya está y es más nuevo que el archivo del dataset (si se volvió a
        # armar el dataset, p. ej. con películas nuevas de los equipos, se recalcula).
        if salida.exists() and salida.stat().st_mtime > archivo.stat().st_mtime:
            continue
        filas = pq.read_table(archivo, columns=["id", "imagen"]).to_pylist()
        vectores = [
            codificador.codificar([datos.abrir_cartel(f["imagen"]) for f in filas[i:i + LOTE]])
            for i in range(0, len(filas), LOTE)
        ]
        np.savez(salida, ids=np.array([f["id"] for f in filas]), vectores=np.concatenate(vectores))
        print(f"{numero}/{len(archivos)} archivos ({len(filas)} carteles)", flush=True)

    guardados = [np.load(partes / f"{a.stem}.npz") for a in archivos]
    np.savez(destino, ids=np.concatenate([g["ids"] for g in guardados]),
             vectores=np.concatenate([g["vectores"] for g in guardados]))
    print(f"Guardados {sum(len(g['ids']) for g in guardados)} vectores en {destino}")


def cargar_split(split: str, embeddings: str = NOMBRE_EMBEDDINGS) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(vectores, etiquetas 0/1, ids) de un split."""
    guardado = np.load(ruta_embeddings(embeddings))
    fila_de = {int(i): n for n, i in enumerate(guardado["ids"])}
    meta = datos.metadata(split)
    meta = meta[meta["id"].isin(fila_de)]
    x = guardado["vectores"][[fila_de[i] for i in meta["id"]]]
    y = np.array([datos.vector_generos(list(g)) for g in meta["generos"]])
    return x, y, meta["id"].tolist()


def calibrar_umbrales(y: np.ndarray, probabilidades: np.ndarray) -> np.ndarray:
    """Para cada género, el umbral (de 0.05 a 0.95) que maximiza su F1."""
    candidatos = np.arange(0.05, 0.96, 0.01)
    umbrales = []
    for g in range(y.shape[1]):
        f1s = [f1_score(y[:, g], probabilidades[:, g] >= u, zero_division=0) for u in candidatos]
        umbrales.append(candidatos[int(np.argmax(f1s))])
    return np.array(umbrales)


def aplicar_umbrales(probabilidades: np.ndarray, umbrales: np.ndarray) -> np.ndarray:
    """Género presente si supera su umbral; si ninguno lo supera, el más probable
    (toda película tiene al menos un género)."""
    prediccion = probabilidades >= umbrales
    vacias = ~prediccion.any(axis=1)
    prediccion[vacias, probabilidades[vacias].argmax(axis=1)] = True
    return prediccion.astype(int)


def entrenar(epocas: int = 200, tasa: float = 1e-3, paciencia: int = 10) -> None:
    config.comprobar_salidas()  # antes de entrenar: si no se puede guardar, que avise ya
    x_train, y_train, _ = cargar_split("train")
    x_val, y_val, _ = cargar_split("val")
    print(f"train: {len(x_train)} | val: {len(x_val)}")

    torch.manual_seed(42)
    capa = construir_cabeza(ARQUITECTURA_BASE, x_train.shape[1], len(config.GENEROS))
    # Pesar los positivos de cada género como negativos/positivos: si no, el
    # modelo aprende a decir "no" a Western siempre y acierta el 99 %.
    positivos = y_train.sum(axis=0)
    pos_weight = torch.tensor((len(y_train) - positivos) / np.maximum(positivos, 1), dtype=torch.float32)
    perdida = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizador = torch.optim.AdamW(capa.parameters(), lr=tasa, weight_decay=1e-4)

    xt, yt = torch.tensor(x_train), torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(x_val)
    mejor, sin_mejorar, mejor_estado = -1.0, 0, None
    for epoca in range(1, epocas + 1):
        capa.train()
        for i in torch.randperm(len(xt)).split(256):
            optimizador.zero_grad()
            perdida(capa(xt[i]), yt[i]).backward()
            optimizador.step()
        capa.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(capa(xv)).numpy()
        f1_macro = f1_score(y_val, aplicar_umbrales(prob_val, np.full(len(config.GENEROS), 0.5)),
                            average="macro", zero_division=0)
        if f1_macro > mejor:
            mejor, sin_mejorar = f1_macro, 0
            mejor_estado = {k: v.clone() for k, v in capa.state_dict().items()}
        else:
            sin_mejorar += 1
            if sin_mejorar >= paciencia:
                break
    print(f"Época {epoca}: mejor F1 macro en val = {mejor:.4f}")

    capa.load_state_dict(mejor_estado)
    with torch.no_grad():
        prob_val = torch.sigmoid(capa(xv)).numpy()
    umbrales = calibrar_umbrales(y_val, prob_val)
    f1_calibrado = f1_score(y_val, aplicar_umbrales(prob_val, umbrales), average="macro", zero_division=0)
    print(f"Con umbrales por género: F1 macro en val = {f1_calibrado:.4f}")

    ruta = config.carpeta_salidas() / MODELO_BASE
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"modelo_clip": MODELO_CLIP, "embeddings": NOMBRE_EMBEDDINGS,
                "arquitectura": ARQUITECTURA_BASE, "generos": config.GENEROS,
                "capa": mejor_estado, "umbrales": umbrales}, ruta)
    print(f"Guardado en {ruta}")


class Clasificador:
    """Interfaz que usan la demo y los demás módulos.

    Clasificador()          el release fijado en RELEASE_CLASIFICADOR (o el base si no hay)
    Clasificador(ruta)      un .pt en particular

    Acepta dos formatos de .pt:
    - el del repo principal (sin `vectores`): CLIP `modelo_clip` + la cabeza, como siempre;
    - el del repo del clasificador (con `vectores`): cada codificador de esa lista
      (p2m/codificadores.py), con sus vectores concatenados en ese orden, + la cabeza."""

    def __init__(self, ruta=None):
        ruta = ruta or ruta_modelo_clasificador()
        guardado = torch.load(ruta, map_location="cpu", weights_only=False)
        self.generos = guardado["generos"]
        self.umbrales = np.asarray(guardado["umbrales"])
        if guardado.get("vectores"):
            from p2m.codificadores import CodificadorVarios

            self.codificador = CodificadorVarios(list(guardado["vectores"]))
            entrada = self.codificador.dim
        else:
            self.codificador = CodificadorCLIP(guardado.get("modelo_clip", MODELO_CLIP))
            entrada = self.codificador.modelo.config.projection_dim
        self.capa = cargar_cabeza(guardado, entrada)

    def probabilidades(self, imagenes) -> np.ndarray:
        vectores = torch.tensor(self.codificador.codificar(imagenes))
        with torch.no_grad():
            return torch.sigmoid(self.capa(vectores)).numpy()

    def predecir(self, imagen) -> dict:
        prob = self.probabilidades([imagen])[0]
        presentes = aplicar_umbrales(prob[None], self.umbrales)[0]
        return {
            "generos": [g for g, p in zip(self.generos, presentes) if p],
            "probabilidades": {g: float(p) for g, p in zip(self.generos, prob)},
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("accion", choices=["embeddings", "entrenar"])
    args = parser.parse_args()
    calcular_embeddings() if args.accion == "embeddings" else entrenar()
