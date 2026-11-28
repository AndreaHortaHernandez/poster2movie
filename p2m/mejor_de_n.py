"""
E7: "mejor de N semillas". Se generan N carteles por película y se queda el que más puntúa un CLIP
DISTINTO al juez (el juez es CLIP ViT-B/32; aquí se usa ViT-L/14), para no inflar la métrica
que se mide al final. Opcionalmente se restan penalizaciones con texto ("cara deformada", "manos
de más", "letras") medidas con el mismo CLIP.

La semilla 0 del grupo es la misma que usa el cartel normal, así que el mejor de N nunca queda
peor que el normal según el puntuador y la comparación en val es limpia.

Uso (Python):
    from p2m import mejor_de_n
    mejor_de_n.activar(n=4, peso_penalizacion=0.5)   # parchea Cartelista.generar
    ...evaluar_val(...)
    mejor_de_n.desactivar()
"""

import numpy as np
from PIL import Image
import torch

from p2m import cartel

MODELO_PUNTUADOR = "openai/clip-vit-large-patch14"
PENALIZACIONES = [
    "a deformed face, distorted eyes, melted face",
    "extra hands, extra fingers, malformed hands",
    "gibberish text, random letters, watermark",
]

_estado = {"original": None, "clip": None, "proc": None, "neg": None}


def _cargar():
    if _estado["clip"] is None:
        from transformers import CLIPModel, CLIPProcessor

        dev = cartel.DISPOSITIVO
        _estado["clip"] = CLIPModel.from_pretrained(MODELO_PUNTUADOR, torch_dtype=torch.float16).to(dev).eval()
        _estado["proc"] = CLIPProcessor.from_pretrained(MODELO_PUNTUADOR)
        _estado["neg"] = _texto(PENALIZACIONES)
    return _estado["clip"], _estado["proc"]


@torch.no_grad()
def _texto(frases):
    clip, proc = _estado["clip"], _estado["proc"]
    t = proc(text=frases, return_tensors="pt", padding=True, truncation=True, max_length=77).to(clip.device)
    v = clip.get_text_features(**t)
    v = v if torch.is_tensor(v) else v.pooler_output  # transformers >= 5 regresa un objeto
    return torch.nn.functional.normalize(v.float(), dim=-1)


@torch.no_grad()
def _imagenes(imagenes):
    clip, proc = _estado["clip"], _estado["proc"]
    x = proc(images=imagenes, return_tensors="pt").to(clip.device)
    v = clip.get_image_features(pixel_values=x["pixel_values"].half())
    v = v if torch.is_tensor(v) else v.pooler_output
    return torch.nn.functional.normalize(v.float(), dim=-1)


def puntuar(imagenes, sinopsis, peso_penalizacion=0.5):
    """Puntuación de cada imagen: similitud con la sinopsis - peso * peor similitud con una penalización."""
    _cargar()
    texto = _texto([sinopsis])[0]
    v = _imagenes(imagenes)
    positiva = (v @ texto).cpu().numpy()
    negativa = (v @ _estado["neg"].T).max(dim=1).values.cpu().numpy()
    return positiva - peso_penalizacion * negativa, positiva, negativa


REGISTRO = []  # una fila por película cuando activar(..., anatomia=True): defectos de la primera y de la elegida


def activar(n=4, peso_penalizacion=0.5, anatomia=False, max_caras=2, intentos_extra=0, usar_clip=True, hires=False):
    """Parchea Cartelista.generar para devolver el mejor de N carteles.

    anatomia=True: además cuenta manos y caras con MediaPipe (p2m.anatomia). Entre las candidatas se
    descartan las que tienen defectos mientras exista una limpia; si ninguna lo está y
    intentos_extra > 0, se generan hasta esas semillas más. La candidata 0 usa la misma semilla que el
    cartel normal, así REGISTRO compara "sin filtro" contra "con filtro" en las mismas películas."""
    if _estado["original"] is not None:
        desactivar()
    original = cartel.Cartelista.generar
    _estado["original"] = original
    REGISTRO.clear()

    def generar_sin_clip(self, generos, resena, semilla=0, titulo=None):
        """Sin puntuador CLIP: genera una semilla tras otra y se queda con la primera que el detector deja
        limpia (hasta n intentos); si ninguna lo está, la de menos defectos (la más temprana en empate).
        Maximizar la similitud CLIP elegía carteles recargados y con sujetos repetidos."""
        from p2m import anatomia as an

        candidatas, cuentas = [], []
        for k in range(n):
            im = original(self, generos, resena, semilla=semilla + 100000 * k)
            candidatas.append(im)
            cuentas.append(an.revisar(im, max_caras))
            if cuentas[-1][0] == 0:
                break
        d = [c[0] for c in cuentas]
        elegida = int(np.argmin(d))
        REGISTRO.append({"semilla": semilla, "candidatas": len(candidatas),
                         "defectos_primera": d[0], "defectos_elegida": d[elegida],
                         "manos_elegida": cuentas[elegida][1], "caras_elegida": cuentas[elegida][2]})
        mejor = candidatas[elegida]
        if hires:
            # La segunda pasada mejora caras pero a veces inventa una mano o una cara de más (en test-500,
            # 3.8 % -> 11.6 % de carteles con defectos detectados). Se acepta solo si no empeora el conteo:
            # hasta 3 intentos con otra semilla y, si ninguno sirve, se queda la imagen sin redibujar.
            d0, final = cuentas[elegida][0], None
            for r in range(3):
                intento = self.mejorar(mejor, generos, resena, semilla + 7919 * r)
                d1 = an.revisar(intento.resize((512, 768)), max_caras)[0]
                if d1 <= d0:
                    final = intento
                    break
            REGISTRO[-1]["hires_descartado"] = final is None
            REGISTRO[-1]["defectos_tras_hires"] = d1 if final is not None else d0
            mejor = final if final is not None else mejor.convert("RGB").resize(cartel.HIRES, Image.LANCZOS)
        if titulo:
            from p2m.cartel_titulo import poner_titulo, quitar_letras

            mejor = poner_titulo(quitar_letras(mejor), titulo, generos)
        return mejor

    def generar_mejor(self, generos, resena, semilla=0, titulo=None):
        candidatas = [original(self, generos, resena, semilla=semilla + 100000 * k) for k in range(n)]
        puntos = list(puntuar(candidatas, resena, peso_penalizacion)[0])
        if anatomia:
            from p2m import anatomia as an

            cuentas = [an.revisar(i, max_caras) for i in candidatas]
            k = n
            while intentos_extra and k < n + intentos_extra and min(c[0] for c in cuentas) > 0:
                nueva = original(self, generos, resena, semilla=semilla + 100000 * k)
                candidatas.append(nueva)
                puntos.append(float(puntuar([nueva], resena, peso_penalizacion)[0][0]))
                cuentas.append(an.revisar(nueva, max_caras))
                k += 1
            d = [c[0] for c in cuentas]
            elegida = an.elegir(puntos, d)
            REGISTRO.append({"semilla": semilla, "candidatas": len(candidatas),
                             "defectos_primera": d[0], "defectos_elegida": d[elegida],
                             "manos_elegida": cuentas[elegida][1], "caras_elegida": cuentas[elegida][2]})
        else:
            elegida = int(np.argmax(puntos))
        mejor = candidatas[elegida]
        if titulo:
            from p2m.cartel_titulo import poner_titulo, quitar_letras

            mejor = poner_titulo(quitar_letras(mejor), titulo, generos)
        return mejor

    cartel.Cartelista.generar = generar_mejor if usar_clip else generar_sin_clip


def resumen_anatomia():
    """Fracción de películas con algún defecto detectado: sin filtro (semilla normal) y con filtro."""
    if not REGISTRO:
        return None
    r = {
        "peliculas": len(REGISTRO),
        "con_defectos_sin_filtro": float(np.mean([x["defectos_primera"] > 0 for x in REGISTRO])),
        "con_defectos_con_filtro": float(np.mean([x["defectos_elegida"] > 0 for x in REGISTRO])),
    }
    if "defectos_tras_hires" in REGISTRO[0]:
        r["con_defectos_tras_hires"] = float(np.mean([x["defectos_tras_hires"] > 0 for x in REGISTRO]))
        r["hires_descartado"] = float(np.mean([x["hires_descartado"] for x in REGISTRO]))
    return r


def desactivar():
    if _estado["original"] is not None:
        cartel.Cartelista.generar = _estado["original"]
        _estado["original"] = None
