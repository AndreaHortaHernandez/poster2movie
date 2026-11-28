"""Rol 7 - Título y letras.
Uso:
    from p2m.cartel_titulo import poner_titulo, quitar_letras
    img2 = poner_titulo(quitar_letras(img_pil), "Título real", ["Horror", "Thriller"])
Las fuentes se descargan solas (Google Fonts, licencia OFL) a DIR_FUENTES.
"""
import os, tempfile, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import matplotlib
DIR_FUENTES = os.environ.get("P2M_FUENTES", os.path.join(tempfile.gettempdir(), "p2m_fuentes"))
# El OCR va en CPU por defecto: así no le quita memoria de GPU al modelo de imagen en la demo.
OCR_GPU = os.environ.get("P2M_OCR_GPU", "0") == "1"
BASE = "https://github.com/google/fonts/raw/main/ofl/"
FUENTE_BASE = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans-Bold.ttf")

# género -> (carpeta/archivo en Google Fonts, color, ¿mayúsculas?)
ESTILOS = {
    "Horror":          ("creepster/Creepster-Regular.ttf",          (200, 20, 20),   True),
    "Science Fiction": ("audiowide/Audiowide-Regular.ttf",          (90, 220, 255),  True),
    "Western":         ("rye/Rye-Regular.ttf",                      (240, 190, 110), True),
    "War":             ("blackopsone/BlackOpsOne-Regular.ttf",      (225, 225, 205), True),
    "Action":          ("anton/Anton-Regular.ttf",                  (255, 150, 30),  True),
    "Romance":         ("greatvibes/GreatVibes-Regular.ttf",        (255, 215, 230), False),
    "Comedy":          ("bangers/Bangers-Regular.ttf",              (255, 220, 40),  True),
    "Animation":       ("pacifico/Pacifico-Regular.ttf",            (255, 235, 90),  False),
    "Family":          ("pacifico/Pacifico-Regular.ttf",            (255, 200, 120), False),
    "Fantasy":         ("cinzeldecorative/CinzelDecorative-Bold.ttf", (240, 210, 120), False),
    "Adventure":       ("cinzeldecorative/CinzelDecorative-Bold.ttf", (235, 200, 110), False),
    "Thriller":        ("bebasneue/BebasNeue-Regular.ttf",          (235, 235, 235), True),
    "Mystery":         ("bebasneue/BebasNeue-Regular.ttf",          (190, 210, 235), True),
    "Crime":           ("bebasneue/BebasNeue-Regular.ttf",          (230, 200, 90),  True),
    "Drama":           ("prata/Prata-Regular.ttf",                  (245, 240, 230), False),
    "History":         ("prata/Prata-Regular.ttf",                  (230, 205, 160), False),
    "Documentary":     ("prata/Prata-Regular.ttf",                  (255, 255, 255), False),
}
COLOR_TEXTO = (255, 247, 232)   # crema
PRIORIDAD = ["Horror", "Science Fiction", "Western", "War", "Action", "Romance", "Comedy",
             "Animation", "Family", "Fantasy", "Adventure", "Thriller", "Mystery", "Crime",
             "Drama", "History", "Documentary"]


def descargar_fuentes():
    """Descarga todas las fuentes a DIR_FUENTES (córrela una vez en Colab)."""
    os.makedirs(DIR_FUENTES, exist_ok=True)
    for ruta, _, _ in ESTILOS.values():
        destino = os.path.join(DIR_FUENTES, os.path.basename(ruta))
        if not os.path.exists(destino):
            try:
                urllib.request.urlretrieve(BASE + ruta, destino)
            except Exception as e:
                print("No se pudo descargar", ruta, e)


def _estilo(generos):
    for g in PRIORIDAD:
        if g in generos:
            return ESTILOS[g]
    return (None, (255, 255, 255), True)


def _fuente(ruta, tam):
    if ruta:
        p = os.path.join(DIR_FUENTES, os.path.basename(ruta))
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, tam)
            except OSError:
                pass
    return ImageFont.truetype(FUENTE_BASE, tam)


def _partir(palabras, n):
    """Reparte las palabras en n líneas lo más parejas posible (por caracteres)."""
    if n == 1 or len(palabras) < n:
        return [" ".join(palabras)]
    mejor, mejor_dif = None, 1e9
    def rec(i, resto, acc):
        nonlocal mejor, mejor_dif
        if resto == 1:
            lineas = acc + [" ".join(palabras[i:])]
            dif = max(len(l) for l in lineas) - min(len(l) for l in lineas)
            if dif < mejor_dif:
                mejor, mejor_dif = lineas, dif
            return
        for j in range(i + 1, len(palabras) - resto + 2):
            rec(j, resto - 1, acc + [" ".join(palabras[i:j])])
    rec(0, n, [])
    return mejor


def _ancho(d, linea, f):
    b = d.textbbox((0, 0), linea, font=f)
    return b[2] - b[0]


def _ajustar(d, texto, ruta, W, H):
    """El tamaño depende del ANCHO: se usa 1 línea si cabe con letra grande (>= 7.5% del alto);
    si no, baja a 2 líneas, luego 3 y 4. Siempre se reduce hasta que quepa: nunca se corta."""
    palabras = texto.split()
    ancho_max, alto_max = W * 0.86, H * 0.30
    tam_max, tam_min = int(H * 0.11), int(H * 0.075)
    mejor = None
    for n in range(1, min(4, len(palabras)) + 1):
        lineas = _partir(palabras, n)
        tam = tam_max
        while tam > 10:
            f = _fuente(ruta, tam)
            if max(_ancho(d, l, f) for l in lineas) <= ancho_max and len(lineas) * tam * 1.18 <= alto_max:
                break
            tam -= 2
        # 1 línea: letra grande (>= 7.5% del alto). 2 líneas: basta con >= 4.5%, así los títulos largos
        # no pasan a 3-4 líneas que tapan las caras del cartel.
        if tam >= (tam_min if n == 1 else int(H * 0.045)):
            return tam, lineas
        if mejor is None or tam > mejor[0]:
            mejor = (tam, lineas)
    return mejor


def _degradado_inferior(img, desde, opac=0.85, rapidez=1.0):
    """Oscurece suavemente desde la fila 'desde' hasta el borde inferior (sin bordes duros)."""
    W, H = img.size
    alto = H - desde
    if alto <= 0:
        return img
    t = np.clip(np.linspace(0, 1, alto) * rapidez, 0, 1)[:, None]
    a = (t * t * (3 - 2 * t)) * opac                      # smoothstep
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr[desde:] *= (1 - a)[:, :, None]
    return Image.fromarray(arr.astype(np.uint8))


def poner_titulo(imagen, titulo, generos):
    descargar_fuentes()   # solo descarga las que faltan
    img = imagen.convert("RGB").copy()
    W, H = img.size
    ruta, _, mayus = _estilo(generos)
    texto = titulo.upper() if mayus else titulo
    tam, lineas = _ajustar(ImageDraw.Draw(img), texto, ruta, W, H)
    f = _fuente(ruta, tam)
    inter = int(tam * 1.18)
    y0 = int(H * 0.955 - inter * len(lineas))

    # degradado oscuro suave detrás del título (más fuerte si el fondo es claro o con detalle)
    zona = np.asarray(img.crop((0, max(0, y0 - 20), W, H)).convert("L"), dtype=float)
    opac = 0.55 + 0.20 * zona.mean() / 255 + 0.10 * min(zona.std(), 80) / 80
    img = _degradado_inferior(img, max(0, y0 - int(H * 0.08)), opac)

    # sombra suave (capa desenfocada) + texto crema
    d0 = ImageDraw.Draw(img)
    pos = []
    for i, l in enumerate(lineas):
        b = d0.textbbox((0, 0), l, font=f)
        pos.append(((W - (b[2] - b[0])) / 2 - b[0], y0 + i * inter, l))
    capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(capa)
    off = max(2, tam // 22)
    for x, y, l in pos:
        sd.text((x + off, y + off), l, font=f, fill=(0, 0, 0, 235))
    capa = capa.filter(ImageFilter.GaussianBlur(max(2, tam / 16)))
    base = img.convert("RGBA")
    base.alpha_composite(capa)
    d = ImageDraw.Draw(base)
    for x, y, l in pos:
        d.text((x, y), l, font=f, fill=COLOR_TEXTO + (255,))
    return base.convert("RGB")


# ---------------- PARTE B: quitar letras falsas ----------------
_lector = None

def _vineta(img, solido, fundido, opac, arriba=False):
    """Franja oscura en el borde: opacidad 'opac' constante en los primeros 'solido' (fracción del alto) y
    luego se desvanece suavemente en 'fundido'. arriba=True la pone en el borde superior."""
    if solido + fundido <= 0:
        return img.convert("RGB")
    W, H = img.size
    a = np.zeros(H, np.float32)
    d = (np.arange(H) / H) if arriba else ((H - 1 - np.arange(H)) / H)      # distancia al borde, en fracción de H
    t = np.clip((d - solido) / max(fundido, 1e-6), 0, 1)
    a = opac * (1 - t * t * (3 - 2 * t))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr *= (1 - a)[:, None, None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _cajas_ocr(arr):
    """Cajas de texto detectadas, con parámetros sensibles (el generador inventa letras muy deformes)."""
    import easyocr
    global _lector
    if _lector is None:
        _lector = easyocr.Reader(["en"], gpu=OCR_GPU)
    return [np.array(c, np.float32) for c, _, _ in _lector.readtext(
        arr, mag_ratio=2.0, text_threshold=0.2, low_text=0.15, link_threshold=0.3)]


def quitar_letras(imagen, cajas=None, profundo=0.84, arriba=0.14, margen=8, radio=5,
                  inf=(0.06, 0.20), opac=1.0):
    """Quita las letras falsas que inventa el generador, SIN tocar el centro de la imagen.
    - Abajo: franja oscura (inf) donde va el título real; el degradado oscurece el texto falso. Las cajas de
      texto que caen en la parte casi negra (cy >= profundo) además se rellenan con inpaint. Más arriba NO:
      el inpaint dejaba parches rectangulares, y oscurecer cajas dejaba bloques negros de borde duro.
    - Arriba: solo renglones de crédito (cajas anchas y bajitas), con inpaint.
    - En medio NO se borra nada: los letreros de la escena (tiendas, calles) son parte del dibujo, y
      repararlos con inpaint dejaba manchas en forma de rombo.
    'cajas' permite pasar las detecciones ya calculadas (para pruebas sin GPU)."""
    import cv2
    arr = np.array(imagen.convert("RGB"))
    H, W = arr.shape[:2]
    if cajas is None:
        cajas = _cajas_ocr(arr)
    mascara = np.zeros((H, W), np.uint8)      # texto abajo y renglones de crédito arriba: se rellenan con inpaint
    for caja in cajas:
        caja = np.asarray(caja, np.float32)
        x0, y0 = caja.min(0); x1, y1 = caja.max(0)
        cy = (y0 + y1) / 2 / H
        a, b = int(max(0, x0 - margen)), int(max(0, y0 - margen))
        c, d = int(min(W, x1 + margen)), int(min(H, y1 + margen))
        # Abajo solo se repara donde la franja oscura ya casi es negra (cy >= profundo): más arriba el
        # inpaint dejaba parches visibles sobre el dibujo y de todos modos el degradado oscurece el texto.
        if cy >= profundo or (cy <= arriba and (x1 - x0) >= 0.2 * W and (y1 - y0) <= 0.04 * H):
            mascara[b:d, a:c] = 255
    if mascara.any():
        arr = cv2.inpaint(arr, mascara, radio, cv2.INPAINT_TELEA)
    res = Image.fromarray(arr)
    return _vineta(res, inf[0], inf[1], opac, arriba=False)
