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
    "Thriller":        ("bebasneue/BebasNeue-Regular.ttf",          (235, 235, 235), True),
    "Mystery":         ("bebasneue/BebasNeue-Regular.ttf",          (190, 210, 235), True),
    "Crime":           ("bebasneue/BebasNeue-Regular.ttf",          (230, 200, 90),  True),
    "Drama":           ("prata/Prata-Regular.ttf",                  (245, 240, 230), False),
    "History":         ("prata/Prata-Regular.ttf",                  (230, 205, 160), False),
    "Documentary":     ("prata/Prata-Regular.ttf",                  (255, 255, 255), False),
}
COLOR_TEXTO = (255, 247, 232)   # crema
PRIORIDAD = ["Horror", "Science Fiction", "Western", "War", "Action", "Romance", "Comedy",
             "Animation", "Family", "Fantasy", "Thriller", "Mystery", "Crime",
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
        if tam >= tam_min:
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

def quitar_letras(imagen, confianza=0.1, dilatar=11, radio=5, franja=0.35,
                  opac_franja=0.92, franja_sup=0.0, opac_sup=0.80):
    """Combina los dos efectos:
    1) OCR + inpaint en TODO el cartel (también abajo), con detección más sensible
       para atrapar texto pequeño o muy deformado.
    2) Degradado oscuro suave sobre la franja inferior (y opcionalmente la superior,
       franja_sup=0.10) que tapa cualquier resto o mancha que deje el inpaint."""
    import cv2, easyocr
    global _lector
    if _lector is None:
        _lector = easyocr.Reader(["en"], gpu=True)
    arr = np.array(imagen.convert("RGB"))
    H = arr.shape[0]
    mascara = np.zeros(arr.shape[:2], np.uint8)
    for caja, texto, conf in _lector.readtext(arr, mag_ratio=2.0, text_threshold=0.4,
                                              low_text=0.25, link_threshold=0.3):
        if conf >= confianza:
            cv2.fillPoly(mascara, [np.array(caja, np.int32)], 255)
    mascara = cv2.dilate(mascara, np.ones((dilatar, dilatar), np.uint8))
    res = Image.fromarray(cv2.inpaint(arr, mascara, radio, cv2.INPAINT_TELEA))
    res = _degradado_inferior(res, int(H * (1 - franja)), opac_franja, rapidez=1.6)
    if franja_sup > 0:
        res = res.transpose(Image.FLIP_TOP_BOTTOM)
        res = _degradado_inferior(res, int(H * (1 - franja_sup)), opac_sup, rapidez=1.6)
        res = res.transpose(Image.FLIP_TOP_BOTTOM)
    return res
