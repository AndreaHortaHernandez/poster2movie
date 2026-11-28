"""
E12: encuesta humana a ciegas.

Protocolo (Andrea): 20 películas x 3 versiones (base, la nuestra y el cartel real), al menos 5 personas,
calificación de 1 a 5 en tres preguntas:
    1. ¿Parece un cartel de cine?            (1 = nada, 5 = mucho)
    2. ¿Corresponde a la historia?           (1 = nada, 5 = mucho)
    3. ¿Hay deformidades? (manos, caras...)  (1 = ninguna, 5 = muchas y evidentes)

Flujo (cada paso en su propia sesión de Colab cuando carga un modelo grande: la RAM no alcanza para dos):
    1. generar("base", carpeta)    -> carteles crudos del sistema base (SD-Turbo)
    2. generar("final", carpeta)   -> carteles crudos de la versión final (SDXL-Turbo + filtro de anatomía, sin selección por CLIP)
    3. armar(carpeta)              -> encuesta.html (autocontenida) y clave_encuesta.csv (NO se comparte)
    4. cada persona abre encuesta.html, responde y descarga su CSV
    5. analizar(carpeta, [csv1, csv2, ...])  -> tabla promedio por versión y diferencias

Las tres versiones se muestran con su título puesto (las generadas pasan por el mismo borrado de letras
falsas + título de la demo; el cartel real ya trae el suyo) y en orden aleatorio por película.
"""

import base64
import csv
import html
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

VERSIONES = ["base", "final", "real"]
ANCHO, ALTO = 384, 576          # tamaño en la encuesta (2:3)
PREGUNTAS = [
    ("parece_cartel", "¿Parece un cartel de cine?", "1 = nada · 5 = mucho"),
    ("corresponde", "¿Corresponde a la historia?", "1 = nada · 5 = mucho"),
    ("deformidades", "¿Hay deformidades? (manos, caras…)", "1 = ninguna · 5 = muchas y evidentes"),
]


def cargar_peliculas(cuantos: int = 20) -> list[dict]:
    from p2m import evaluar_cartel_val as v

    return v.muestra_val(cuantos)


# --------------------------------------------------------------------------- 1 y 2: generar

def _instruccion_base(generos, resena):
    """Prompt del sistema base original (commit ff1bac3), para reproducir el cartel base."""
    import re

    primera = re.split(r"(?<=[.!?])\s", resena.strip(), maxsplit=1)[0]
    primera = " ".join(primera.split()[:35])
    tipo = "/".join(g.lower() for g in generos[:2]) or "feature"
    return f"Cinematic {tipo} movie poster, professional key art, dramatic lighting, no text: {primera}"


def generar(version: str, carpeta, cuantos: int = 20):
    """Genera y guarda los carteles crudos (sin título) de 'base' o 'final'."""
    from p2m import cartel

    assert version in ("base", "final"), "version debe ser 'base' o 'final'"
    carpeta = Path(carpeta)
    crudas = carpeta / "crudas"
    crudas.mkdir(parents=True, exist_ok=True)
    peliculas = cargar_peliculas(cuantos)

    instruccion_original = cartel.instruccion
    ancho, alto = cartel.ANCHO, cartel.ALTO
    try:
        cartel.ANCHO, cartel.ALTO = 512, 768
        if version == "base":
            cartel.instruccion = _instruccion_base
            cartelista = cartel.Cartelista("stabilityai/sd-turbo")
        else:
            from p2m import mejor_de_n

            cartelista = cartel.Cartelista("stabilityai/sdxl-turbo")
            mejor_de_n.activar(n=6, anatomia=True, max_caras=2, usar_clip=False, hires=True)
        for i, f in enumerate(peliculas, start=1):
            imagen = cartelista.generar(list(f["generos"]), f["overview"], semilla=f["id"])
            imagen.save(crudas / f"{version}_{f['id']}.png")
            if i % 5 == 0:
                print(f"{version}: {i}/{len(peliculas)}", flush=True)
    finally:
        cartel.instruccion = instruccion_original
        cartel.ANCHO, cartel.ALTO = ancho, alto
        if version == "final":
            from p2m import mejor_de_n

            mejor_de_n.desactivar()
    print(f"Listo: {len(peliculas)} carteles '{version}' en {crudas}")


# --------------------------------------------------------------------------- 3: armar la encuesta

def _recorte_2_3(imagen: Image.Image) -> Image.Image:
    """Recorte centrado a proporción 2:3 y tamaño de la encuesta (sin deformar)."""
    w, h = imagen.size
    if w / h > 2 / 3:
        nuevo_w = round(h * 2 / 3)
        izq = (w - nuevo_w) // 2
        imagen = imagen.crop((izq, 0, izq + nuevo_w, h))
    else:
        nuevo_h = round(w * 3 / 2)
        sup = (h - nuevo_h) // 2
        imagen = imagen.crop((0, sup, w, sup + nuevo_h))
    return imagen.resize((ANCHO, ALTO), Image.LANCZOS)


def _jpeg_base64(imagen: Image.Image) -> str:
    buf = io.BytesIO()
    imagen.convert("RGB").save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _sinopsis_corta(texto: str, limite: int = 380) -> str:
    texto = " ".join(texto.split())
    if len(texto) <= limite:
        return texto
    corte = texto[:limite].rsplit(" ", 1)[0]
    return corte + "…"


def armar(carpeta, cuantos: int = 20, con_titulo: bool = True, semilla: int = 7, peliculas=None):
    """Construye encuesta.html y clave_encuesta.csv a partir de los carteles crudos."""
    carpeta = Path(carpeta)
    peliculas = peliculas if peliculas is not None else cargar_peliculas(cuantos)
    if con_titulo:
        from p2m.cartel_titulo import poner_titulo, quitar_letras
    else:
        quitar_letras = poner_titulo = None
    azar = random.Random(semilla)
    tarjetas, clave = [], []
    for k, f in enumerate(peliculas, start=1):
        imagenes = {"real": _recorte_2_3(Image.open(io.BytesIO(f["imagen"])).convert("RGB"))}
        for version in ("base", "final"):
            crudo = Image.open(carpeta / "crudas" / f"{version}_{f['id']}.png").convert("RGB")
            if con_titulo:
                crudo = poner_titulo(quitar_letras(crudo), f["title"], list(f["generos"]))
            imagenes[version] = _recorte_2_3(crudo)
        orden = azar.sample(VERSIONES, len(VERSIONES))
        letras = ["A", "B", "C"]
        clave.append({"pelicula": k, "id": f["id"], "titulo": f["title"],
                      **{letra: version for letra, version in zip(letras, orden)}})
        tarjetas.append({
            "k": k, "titulo": f["title"], "generos": ", ".join(f["generos"]),
            "sinopsis": _sinopsis_corta(f["overview"]),
            "imagenes": [(letra, _jpeg_base64(imagenes[version])) for letra, version in zip(letras, orden)],
        })

    pd.DataFrame(clave).to_csv(carpeta / "clave_encuesta.csv", index=False)
    (carpeta / "encuesta.html").write_text(_html(tarjetas), encoding="utf-8")
    print(f"Encuesta lista: {carpeta / 'encuesta.html'} ({len(tarjetas)} películas). "
          f"La clave ({carpeta / 'clave_encuesta.csv'}) NO se comparte con quien responde.")


def _html(tarjetas) -> str:
    bloques = []
    for t in tarjetas:
        columnas = []
        for letra, b64 in t["imagenes"]:
            filas = []
            for clave_pregunta, texto, escala in PREGUNTAS:
                opciones = "".join(
                    f'<label><input type="radio" name="p{t["k"]}_{letra}_{clave_pregunta}" value="{v}">{v}</label>'
                    for v in range(1, 6))
                filas.append(
                    f'<div class="pregunta"><div class="texto">{html.escape(texto)}'
                    f'<span class="escala">{html.escape(escala)}</span></div>'
                    f'<div class="opciones">{opciones}</div></div>')
            columnas.append(
                f'<div class="version"><div class="letra">{letra}</div>'
                f'<img src="data:image/jpeg;base64,{b64}" alt="Cartel {letra}">{"".join(filas)}</div>')
        bloques.append(
            f'<section class="pelicula" data-k="{t["k"]}">'
            f'<h2>Película {t["k"]}: {html.escape(t["titulo"])} <small>({html.escape(t["generos"])})</small></h2>'
            f'<p class="sinopsis">{html.escape(t["sinopsis"])}</p>'
            f'<div class="versiones">{"".join(columnas)}</div></section>')

    total = len(tarjetas) * 3 * len(PREGUNTAS)
    preguntas_js = ",".join(f'"{q[0]}"' for q in PREGUNTAS)
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Encuesta de carteles</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1250px;margin:0 auto;padding:16px;background:#fafafa;color:#222}}
header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 0;z-index:5}}
h1{{font-size:20px;margin:0 0 6px}} h2{{font-size:17px;margin:0 0 4px}} small{{font-weight:400;color:#666}}
.sinopsis{{margin:2px 0 10px;color:#444;font-size:14px}}
.pelicula{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:14px;margin:18px 0}}
.versiones{{display:flex;gap:14px;flex-wrap:wrap}}
.version{{flex:1 1 300px;min-width:280px;max-width:380px}}
.version img{{width:100%;border-radius:6px;display:block;border:1px solid #ccc}}
.letra{{font-weight:700;font-size:18px;margin-bottom:4px}}
.pregunta{{margin-top:8px;font-size:13px}} .escala{{display:block;color:#777;font-size:11px}}
.opciones{{display:flex;gap:6px;margin-top:2px}} .opciones label{{border:1px solid #bbb;border-radius:6px;padding:3px 9px;cursor:pointer}}
.opciones label:has(input:checked){{background:#2563eb;color:#fff;border-color:#2563eb}}
.opciones input{{display:none}}
button{{background:#2563eb;color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:14px;cursor:pointer}}
#estado{{margin-left:10px;font-size:13px;color:#555}}
</style></head><body>
<header>
<h1>Encuesta de carteles de cine</h1>
<div>Tu nombre o apodo: <input id="evaluador" type="text" size="18">
<button id="descargar">Descargar mis respuestas (CSV)</button><span id="estado"></span></div>
<div style="font-size:13px;color:#555;margin-top:4px">Para cada película hay tres carteles (A, B, C) en orden al azar.
Califica cada uno de 1 a 5 en las tres preguntas. Toma unos 10 a 15 minutos.</div>
</header>
{"".join(bloques)}
<script>
const TOTAL = {total};
const PREGUNTAS = [{preguntas_js}];
function contar() {{
  return document.querySelectorAll('input[type=radio]:checked').length;
}}
function estado() {{
  document.getElementById('estado').textContent = contar() + ' de ' + TOTAL + ' respuestas';
}}
document.addEventListener('change', estado);
estado();
function construirCSV() {{
  const nombre = (document.getElementById('evaluador').value || 'anonimo').replace(/[",\\n]/g, ' ');
  const filas = ['evaluador,pelicula,letra,' + PREGUNTAS.join(',')];
  document.querySelectorAll('section.pelicula').forEach(sec => {{
    const k = sec.dataset.k;
    ['A', 'B', 'C'].forEach(letra => {{
      const vals = PREGUNTAS.map(q => {{
        const sel = sec.querySelector('input[name="p' + k + '_' + letra + '_' + q + '"]:checked');
        return sel ? sel.value : '';
      }});
      filas.push([nombre, k, letra].concat(vals).join(','));
    }});
  }});
  return filas.join('\\n') + '\\n';
}}
document.getElementById('descargar').addEventListener('click', () => {{
  const faltan = TOTAL - contar();
  if (faltan > 0 && !window.__avisado) {{
    window.__avisado = true;
    document.getElementById('estado').textContent = 'Te faltan ' + faltan + ' respuestas. Si aun así quieres descargar, vuelve a pulsar el botón.';
    return;
  }}
  const nombre = (document.getElementById('evaluador').value || 'anonimo').replace(/[^A-Za-z0-9_-]/g, '_');
  const blob = new Blob([construirCSV()], {{type: 'text/csv;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'respuestas_' + nombre + '.csv';
  document.body.appendChild(a); a.click(); a.remove();
}});
</script></body></html>"""


# --------------------------------------------------------------------------- 5: analizar

def analizar(carpeta, archivos_csv, guardar=True, remuestreos=10000):
    """Promedio por versión y pregunta, y diferencia final - base con intervalo (bootstrap por película)."""
    carpeta = Path(carpeta)
    clave = pd.read_csv(carpeta / "clave_encuesta.csv")
    largo = clave.melt(id_vars=["pelicula"], value_vars=["A", "B", "C"], var_name="letra", value_name="version")
    crudo = pd.concat([pd.read_csv(a) for a in archivos_csv], ignore_index=True)
    datos = crudo.merge(largo, on=["pelicula", "letra"], how="left")
    columnas = [p[0] for p in PREGUNTAS]
    datos[columnas] = datos[columnas].apply(pd.to_numeric, errors="coerce")
    n_eval = datos["evaluador"].nunique()

    tabla = datos.groupby("version")[columnas].agg(["mean", "std", "count"]).round(2)
    print(f"Evaluadores: {n_eval}   (el protocolo pide al menos 5)")
    print(tabla.to_string())

    por_pelicula = datos.groupby(["pelicula", "version"])[columnas].mean().reset_index()
    azar = np.random.default_rng(0)
    filas = []
    for q in columnas:
        ancho = por_pelicula.pivot(index="pelicula", columns="version", values=q)
        for a, b in [("final", "base"), ("final", "real")]:
            d = (ancho[a] - ancho[b]).dropna().values
            if len(d) == 0:
                continue
            medias = [azar.choice(d, len(d)).mean() for _ in range(remuestreos)]
            lo, hi = np.percentile(medias, [2.5, 97.5])
            filas.append({"pregunta": q, "comparacion": f"{a} - {b}", "diferencia": round(d.mean(), 2),
                          "IC95_bajo": round(lo, 2), "IC95_alto": round(hi, 2), "peliculas": len(d)})
    diferencias = pd.DataFrame(filas)
    print(diferencias.to_string(index=False))
    if guardar:
        tabla.to_csv(carpeta / "encuesta_promedios.csv")
        diferencias.to_csv(carpeta / "encuesta_diferencias.csv", index=False)
        print(f"Guardado en {carpeta}")
    return tabla, diferencias
