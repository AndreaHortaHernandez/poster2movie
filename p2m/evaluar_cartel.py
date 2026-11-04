"""
Evaluación del módulo de cartel en una muestra FIJA del split de test.

Para cada película se genera un cartel a partir de sus géneros y su sinopsis
REALES (así se mide solo el cartel, sin errores de los otros módulos) y se mide:

- CLIPScore cartel <-> sinopsis: 100 x similitud CLIP entre la imagen y el
  texto. Se compara con el CLIPScore del cartel REAL de la misma película
  (referencia de "qué tan bien lo hace un cartel de verdad").
- Género: se le pasa el cartel generado al clasificador (módulo 1) y se mide
  si detecta los géneros pedidos (F1 por película). También se mide con el
  cartel real como referencia.

Además guarda una lámina con carteles reales vs generados para revisarlos a ojo.

Uso:
    python -m p2m.evaluar_cartel                          # 100 películas, modelo base
    python -m p2m.evaluar_cartel --cuantos 30 --nombre equipoX_sdxl --modelo stabilityai/sdxl-turbo

Guarda en <carpeta de salidas>/resultados/cartel/<nombre>_metricas.csv, _ejemplos.csv y _lamina.jpg.
"""

import argparse
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

from p2m import config, datos
from p2m.cartel import MODELO_CARTEL, Cartelista
from p2m.evaluar_resena import muestra_test

RESULTADOS = config.carpeta_salidas() / "resultados" / "cartel"


def lamina(reales, generados, ruta, cuantos=6) -> None:
    """Fila de arriba: carteles reales; fila de abajo: generados."""
    ancho, alto = 256, 384
    hoja = Image.new("RGB", (ancho * cuantos, alto * 2), "white")
    for i, (real, generado) in enumerate(zip(reales[:cuantos], generados[:cuantos])):
        hoja.paste(real.resize((ancho, alto)), (i * ancho, 0))
        hoja.paste(generado.resize((ancho, alto)), (i * ancho, alto))
    hoja.save(ruta, quality=90)


def evaluar(modelo=MODELO_CARTEL, nombre="base", cuantos=100) -> pd.DataFrame:
    from p2m.clasificador import Clasificador

    peliculas = muestra_test(cuantos)
    cartelista = Cartelista(modelo)
    generados, inicio = [], time.time()
    for n, f in enumerate(peliculas, start=1):
        generados.append(cartelista.generar(list(f["generos"]), f["overview"], semilla=f["id"]))
        if n % 20 == 0:
            print(f"{n}/{len(peliculas)} ({time.time() - inicio:.0f}s)", flush=True)
    del cartelista

    reales = [datos.abrir_cartel(f["imagen"]) for f in peliculas]
    clasificador = Clasificador()
    clip = clasificador.codificador
    texto = clip.codificar_texto([f["overview"] for f in peliculas])
    clip_generado = 100 * np.clip((clip.codificar(generados) * texto).sum(axis=1), 0, None)
    clip_real = 100 * np.clip((clip.codificar(reales) * texto).sum(axis=1), 0, None)

    esperados = np.array([datos.vector_generos(list(f["generos"])) for f in peliculas])

    def generos_detectados(imagenes):
        return np.array([datos.vector_generos(clasificador.predecir(im)["generos"]) for im in imagenes])

    detectados_gen, detectados_real = generos_detectados(generados), generos_detectados(reales)

    tabla = pd.DataFrame([{
        "modelo": nombre, "generador": str(modelo), "peliculas": len(peliculas),
        "CLIPScore_generado": clip_generado.mean(), "CLIPScore_real": clip_real.mean(),
        "F1_genero_generado": f1_score(esperados, detectados_gen, average="samples", zero_division=0),
        "F1_genero_real": f1_score(esperados, detectados_real, average="samples", zero_division=0),
    }])
    ejemplos = pd.DataFrame({
        "id": [f["id"] for f in peliculas], "titulo": [f["title"] for f in peliculas],
        "generos": [", ".join(f["generos"]) for f in peliculas],
        "CLIPScore_generado": clip_generado, "CLIPScore_real": clip_real,
    })

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RESULTADOS / f"{nombre}_metricas.csv", index=False)
    ejemplos.to_csv(RESULTADOS / f"{nombre}_ejemplos.csv", index=False)
    lamina(reales, generados, RESULTADOS / f"{nombre}_lamina.jpg")
    print(tabla.to_string(index=False, float_format="{:.3f}".format))
    return tabla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default=MODELO_CARTEL)
    parser.add_argument("--nombre", default="base")
    parser.add_argument("--cuantos", type=int, default=100)
    args = parser.parse_args()
    evaluar(args.modelo, args.nombre, args.cuantos)
