# Poster2Movie — Dataset

Sistema de ML que recibe el **cartel de una película** y genera:
**género + reseña + nuevo cartel**.

Este repositorio junta el dataset base
[`skvarre/movie_posters-100k`](https://huggingface.co/datasets/skvarre/movie_posters-100k)
(~95,300 películas, NO se sube aquí) con las películas que agrega cada equipo.

## Géneros por equipo

| Equipo | Carpeta | Géneros (ids TMDB) |
|---|---|---|
| Equipo 1 | `contribuciones/equipo1` | _(llenar)_ |
| Equipo 2 | `contribuciones/equipo2` | _(llenar)_ |
| Equipo 3 | `contribuciones/equipo3` | _(llenar)_ |
| Equipo 4 | `contribuciones/equipo4` | _(llenar)_ |

Ids de género de TMDB: Action=28, Adventure=12, Animation=16, Comedy=35, Crime=80,
Documentary=99, Drama=18, Family=10751, Fantasy=14, History=36, Horror=27, Music=10402,
Mystery=9648, Romance=10749, Science Fiction=878, TV Movie=10770, Thriller=53, War=10752, Western=37.

**Regla de asignación:** una película pertenece al equipo de su **primer género** en TMDB.
El script lo aplica solo, así nadie duplica películas.

## Estructura

```
contribuciones/
  equipoN/
    metadata.csv          # una fila por película
    posters/{id}.jpg      # cartel, nombrado con el id de TMDB
data/ids_base.txt         # ids que ya están en el dataset base (para no repetir)
scripts/
  descargar_tmdb.py       # descarga películas nuevas de TMDB
  validar.py              # revisa que todo esté bien antes de subir
```

### Columnas de `metadata.csv`

Las mismas que el dataset base + `coleccion` + `equipo`:

`id, title, genres, overview, popularity, release_date, budget, revenue, tagline, original_language, runtime, coleccion, equipo`

- `genres` es JSON: `[{"id": 27, "name": "Horror"}, {"id": 53, "name": "Thriller"}]`
- `coleccion` es el id de TMDB de la franquicia (`belongs_to_collection`), vacío si no tiene.
  Sirve para que en el split train/test las películas de una misma saga queden juntas
  (si no, el modelo "acierta" porque ya vio un cartel casi igual de la secuela).
  Es normal que películas de una misma franquicia tengan géneros distintos en TMDB; no se corrigen a mano.
  Si ya tenías un `metadata.csv` sin esta columna, corre `descargar_tmdb.py` otra vez (puede ser con `--max 0`) y la agrega sola.
- `overview` (sinopsis) en **inglés**, mínimo 20 palabras.
- Cartel en JPG, tamaño `w500` de TMDB.

## Pasos para cada equipo

### 1. Una sola vez

1. Crear cuenta en [themoviedb.org](https://www.themoviedb.org/signup) → *Settings → API* → pedir una API key (gratis, uso educativo).
2. Clonar el repo e instalar dependencias:
   ```bash
   git clone <URL_DEL_REPO>
   cd poster2movie
   pip install -r requirements.txt
   ```
3. Guardar la API key en la terminal:
   ```bash
   # Windows (PowerShell)
   $env:TMDB_API_KEY = "tu_api_key"
   # Mac / Linux
   export TMDB_API_KEY="tu_api_key"
   ```

### 2. Descargar películas

```bash
git pull                                   # siempre antes de descargar
python scripts/descargar_tmdb.py --equipo equipo2 --generos 27 53 80 9648 --max 300
```

Ideas para agregar películas que **sí aportan** (no repetidas del dataset base):

```bash
# Estrenos recientes
python scripts/descargar_tmdb.py --equipo equipo2 --generos 27 53 --desde 2023-01-01 --max 200
# Cine mexicano
python scripts/descargar_tmdb.py --equipo equipo2 --generos 27 53 --pais MX --min-votos 5 --max 100
# Cine en español
python scripts/descargar_tmdb.py --equipo equipo2 --generos 27 53 --idioma es --min-votos 5 --max 100
```

Se puede interrumpir con `Ctrl+C` y volver a correr: continúa donde se quedó.

### 3. Validar

```bash
python scripts/validar.py equipo2
```

No subir nada hasta que diga **OK**.

### 4. Subir

```bash
git add contribuciones/equipo2
git commit -m "equipo2: +300 películas (terror/thriller)"
git pull --rebase
git push
```

## Reglas

- Cada equipo **solo toca su carpeta** `contribuciones/equipoN/`. Así no hay conflictos en git.
- No editar a mano `metadata.csv` (si hay que borrar una película, borrar su fila **y** su imagen).
- No subir el dataset base ni archivos pesados que no sean carteles.
- Meta por equipo: **500–1,000 películas**, con al menos ~100 por cada género asignado.
