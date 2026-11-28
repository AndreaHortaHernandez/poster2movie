"""
Detector de defectos anatómicos para los carteles (E7: "penalización si MediaPipe detecta manos de
más o caras deformes").

Cuenta manos y caras con MediaPipe (API Tasks) y define un número de "defectos":

    defectos = manos de más + caras de más
    manos de más = max(0, manos - 2 * max(1, caras))       # una persona tiene 2 manos
    caras de más = max(0, caras - max_caras)               # sujeto duplicado / multitud inventada

Sirve para dos cosas:
  1. Filtrar: en el "mejor de N" se descartan los carteles con defectos mientras exista uno limpio.
  2. Medir: tasa de carteles con algún defecto detectado, antes y después del filtro.

Existe además un detector de personas, bicicletas y motos repetidas (EfficientDet de MediaPipe; `revisar(...,
objetos=True)`), APAGADO por defecto: probado el 27 nov sobre los 60 carteles de la encuesta 3, marcó 14 de 20
carteles finales (casi todos carteles de grupo legítimos), no vio el jinete fusionado ni las bicicletas
apiladas que sí eran defectos, y solo acertó 2 casos reales. Se reporta como intento fallido.

LÍMITES (para el reporte): solo ve lo que MediaPipe detecta. Manos pequeñas, borrosas o muy
estilizadas pueden pasar sin detectarse, y una cara real muy chica puede no contarse. Por eso el
filtro reduce los defectos detectables, no garantiza cero; lo que queda se mide con la encuesta.

Los modelos (.task / .tflite) se descargan una vez desde storage.googleapis.com a CARPETA.
"""

import os
import urllib.request

import numpy as np

CARPETA = os.environ.get("P2M_MODELOS_MP", "/tmp/modelos_mediapipe")
MAX_CARAS = 2
CONFIANZA_CARA = 0.3   # más baja que el 0.5 por defecto: detecta también la cara borrosa de un doble

_BASE = "https://storage.googleapis.com/mediapipe-models"
MODELOS = {
    "mano": ("hand_landmarker.task",
             [f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
              f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"]),
    "objeto": ("efficientdet.tflite",
               [f"{_BASE}/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite",
                f"{_BASE}/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite",
                f"{_BASE}/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"]),
    "cara": ("blaze_face_short_range.tflite",
             [f"{_BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
              f"{_BASE}/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"]),
}


def _descargar(clave):
    nombre, urls = MODELOS[clave]
    os.makedirs(CARPETA, exist_ok=True)
    ruta = os.path.join(CARPETA, nombre)
    if os.path.exists(ruta) and os.path.getsize(ruta) > 10_000:
        return ruta
    ultimo = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, ruta)
            if os.path.getsize(ruta) > 10_000:
                return ruta
        except Exception as e:  # noqa: BLE001
            ultimo = e
    raise RuntimeError(
        f"No pude descargar el modelo de MediaPipe '{nombre}' ({ultimo}). Descárgalo a mano de "
        f"{urls[0]} y ponlo en {CARPETA}/")


def defectos(manos: int, caras: int, max_caras: int = MAX_CARAS) -> int:
    manos_de_mas = max(0, manos - 2 * max(1, caras))
    caras_de_mas = max(0, caras - max_caras)
    return manos_de_mas + caras_de_mas


def elegir(puntos, num_defectos):
    """Índice de la mejor candidata: entre las de menos defectos, la de mayor puntuación."""
    puntos = list(map(float, puntos))
    num_defectos = list(map(int, num_defectos))
    minimo = min(num_defectos)
    candidatas = [i for i, d in enumerate(num_defectos) if d == minimo]
    return max(candidatas, key=lambda i: puntos[i])


def dobles(cajas) -> int:
    """Pares de caras apiladas en vertical (una encima de la otra, casi en la misma columna y de tamaño
    parecido): es la firma típica de un sujeto duplicado. cajas: lista de (cx, cy, ancho, alto)."""
    n = 0
    for i in range(len(cajas)):
        for j in range(i + 1, len(cajas)):
            xi, yi, wi, hi = cajas[i]
            xj, yj, wj, hj = cajas[j]
            parecidas = 0.5 <= wi / max(wj, 1e-6) <= 2.0
            misma_columna = abs(xi - xj) < 0.5 * max(wi, wj)
            una_sobre_otra = abs(yi - yj) > 0.6 * max(hi, hj)
            if parecidas and misma_columna and una_sobre_otra:
                n += 1
    return n


CLASES_OBJETO = ("person", "bicycle", "motorcycle")   # lo que el generador suele duplicar o fusionar
CONFIANZA_OBJETO = 0.3
MAX_VEHICULOS = 2                                     # más de 2 motos (o de 2 bicis) en un cartel = repetidas


def repetidos(objetos) -> int:
    """Objetos duplicados o apilados que no son caras: personas, bicicletas y motos.
    objetos: lista de (clase, cx, cy, ancho, alto) en fracción de la imagen.
    - pares de la misma clase apilados en vertical (misma regla que para las caras);
    - más de MAX_VEHICULOS bicicletas o motos."""
    n = 0
    for clase in CLASES_OBJETO:
        cajas = [(cx, cy, w, h) for c, cx, cy, w, h in objetos if c == clase]
        n += dobles(cajas)
        if clase != "person":
            n += max(0, len(cajas) - MAX_VEHICULOS)
    return n


class Detector:
    """Cuenta manos y caras en una imagen PIL. Carga MediaPipe la primera vez que se usa."""

    def __init__(self):
        self._mano = None
        self._cara = None
        self._objeto = None

    def _cargar(self):
        if self._mano is not None:
            return
        import mediapipe as mp  # noqa: F401  (se usa en contar)
        from mediapipe.tasks.python import BaseOptions, vision

        self._mano = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_descargar("mano")),
            num_hands=8, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5))
        self._cara = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=_descargar("cara")),
            min_detection_confidence=CONFIANZA_CARA))
        self._objeto = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=_descargar("objeto")),
            max_results=25, score_threshold=CONFIANZA_OBJETO))

    def objetos(self, imagen):
        """-> [(clase, cx, cy, ancho, alto)] de personas, bicicletas y motos, en fracción de la imagen."""
        import mediapipe as mp

        self._cargar()
        datos = np.ascontiguousarray(np.asarray(imagen.convert("RGB")))
        W, H = imagen.size
        res = self._objeto.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=datos))
        salida = []
        for d in res.detections:
            clase = d.categories[0].category_name
            if clase in CLASES_OBJETO:
                b = d.bounding_box
                salida.append((clase, (b.origin_x + b.width / 2) / W, (b.origin_y + b.height / 2) / H,
                               b.width / W, b.height / H))
        return salida

    def analizar(self, imagen):
        """-> (manos, cajas_de_caras) con cajas = [(cx, cy, ancho, alto)] en fracción del tamaño de la imagen."""
        import mediapipe as mp

        self._cargar()
        datos = np.ascontiguousarray(np.asarray(imagen.convert("RGB")))
        W, H = imagen.size
        mp_imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=datos)
        manos = len(self._mano.detect(mp_imagen).hand_landmarks)
        cajas = []
        for d in self._cara.detect(mp_imagen).detections:
            b = d.bounding_box
            cajas.append(((b.origin_x + b.width / 2) / W, (b.origin_y + b.height / 2) / H, b.width / W, b.height / H))
        return manos, cajas

    def contar(self, imagen):
        """-> (manos, caras)"""
        manos, cajas = self.analizar(imagen)
        return manos, len(cajas)


_detector = Detector()


def contar(imagen):
    return _detector.contar(imagen)


def revisar(imagen, max_caras: int = MAX_CARAS, objetos: bool = False):
    """-> (defectos, manos, caras): manos de más, caras de más, caras duplicadas apiladas y, si
    objetos=True, personas/bicicletas/motos repetidas."""
    manos, cajas = _detector.analizar(imagen)
    total = defectos(manos, len(cajas), max_caras) + dobles(cajas)
    if objetos:
        total += repetidos(_detector.objetos(imagen))
    return total, manos, len(cajas)


def tasa_de_defectos(imagenes, max_caras: int = MAX_CARAS):
    """Para una lista de imágenes: fracción con algún defecto, y medias de manos y caras detectadas."""
    cuentas = [contar(i) for i in imagenes]
    d = [defectos(m, c, max_caras) for m, c in cuentas]
    return {
        "con_defectos": float(np.mean([x > 0 for x in d])),
        "manos_media": float(np.mean([m for m, _ in cuentas])),
        "caras_media": float(np.mean([c for _, c in cuentas])),
        "n": len(d),
    }
