"""
Demo de punta a punta: subes un cartel y regresa géneros + reseña + cartel nuevo.

    python -m p2m.demo          # en Colab imprime un link público (gradio.live) que dura 72 h

Usa los tres módulos tal cual: cuando un equipo mejora el suyo y lo sube al
repo, la demo lo usa sin cambiar nada aquí.
"""

import gradio as gr

from p2m.cartel import Cartelista
from p2m.clasificador import Clasificador
from p2m.config import GENEROS_ES
from p2m.resena import Resenador

print("Cargando los tres modelos (la primera vez se descargan, ~8 GB)...")
clasificador = Clasificador()
resenador = Resenador()
cartelista = Cartelista()


def procesar(cartel, semilla):
    if cartel is None:
        raise gr.Error("Sube un cartel primero.")
    cartel = cartel.convert("RGB")
    clasificacion = clasificador.predecir(cartel)
    generos = clasificacion["generos"]
    probabilidades = {GENEROS_ES[g]: p for g, p in clasificacion["probabilidades"].items()}
    resena = resenador.generar(cartel, generos)
    nuevo = cartelista.generar(generos, resena, semilla=int(semilla))
    return ", ".join(GENEROS_ES[g] for g in generos), probabilidades, resena, nuevo


with gr.Blocks(title="Poster2Movie") as demo:
    gr.Markdown("# Poster2Movie\nSube el cartel de una película: el sistema adivina sus géneros, "
                "escribe una sinopsis y genera un cartel nuevo.")
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="pil", label="Cartel")
            semilla = gr.Number(value=0, precision=0, label="Semilla del cartel nuevo (cámbiala para otra versión)")
            boton = gr.Button("Analizar", variant="primary")
        with gr.Column():
            generos = gr.Textbox(label="Géneros")
            probabilidades = gr.Label(num_top_classes=5, label="Probabilidad por género")
            resena = gr.Textbox(label="Sinopsis generada (inglés)", lines=5)
        nuevo = gr.Image(label="Cartel nuevo")
    boton.click(procesar, [entrada, semilla], [generos, probabilidades, resena, nuevo])

if __name__ == "__main__":
    demo.launch(share=True)
