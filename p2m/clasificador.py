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

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import CLIPModel, CLIPProcessor

from p2m import config, datos

MODELO_CLIP = "openai/clip-vit-base-patch32"
RUTA_MODELO = config.carpeta_datos() / "modelos" / "clasificador_base.pt"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
LOTE = 64


def ruta_embeddings():
    return datos.carpeta() / "embeddings_clip_b32.npz"


class CodificadorCLIP:
    def __init__(self):
        self.procesador = CLIPProcessor.from_pretrained(MODELO_CLIP)
        self.modelo = CLIPModel.from_pretrained(MODELO_CLIP).to(DISPOSITIVO).eval()

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
    """Vector CLIP de cada cartel del dataset, guardado junto al dataset."""
    codificador = CodificadorCLIP()
    ids, vectores, lote = [], [], []

    def vaciar():
        vectores.append(codificador.codificar([datos.abrir_cartel(b) for _, b in lote]))
        ids.extend(i for i, _ in lote)
        lote.clear()

    for n, fila in enumerate(datos.peliculas(columnas=["id", "imagen"]), start=1):
        lote.append((fila["id"], fila["imagen"]))
        if len(lote) == LOTE:
            vaciar()
        if n % 5000 == 0:
            print(f"{n} carteles", flush=True)
    if lote:
        vaciar()
    np.savez(ruta_embeddings(), ids=np.array(ids), vectores=np.concatenate(vectores))
    print(f"Guardados {len(ids)} vectores en {ruta_embeddings()}")


def cargar_split(split: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(vectores, etiquetas 0/1, ids) de un split."""
    guardado = np.load(ruta_embeddings())
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
    x_train, y_train, _ = cargar_split("train")
    x_val, y_val, _ = cargar_split("val")
    print(f"train: {len(x_train)} | val: {len(x_val)}")

    torch.manual_seed(42)
    capa = torch.nn.Linear(x_train.shape[1], len(config.GENEROS))
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

    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"modelo_clip": MODELO_CLIP, "generos": config.GENEROS,
                "capa": mejor_estado, "umbrales": umbrales}, RUTA_MODELO)
    print(f"Guardado en {RUTA_MODELO}")


class Clasificador:
    """Interfaz que usan la demo y los demás módulos."""

    def __init__(self, ruta=RUTA_MODELO):
        guardado = torch.load(ruta, map_location="cpu", weights_only=False)
        self.generos = guardado["generos"]
        self.umbrales = np.asarray(guardado["umbrales"])
        self.capa = torch.nn.Linear(guardado["capa"]["weight"].shape[1], len(self.generos))
        self.capa.load_state_dict(guardado["capa"])
        self.capa.eval()
        self.codificador = CodificadorCLIP()

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
