Hola a todos 👋 Para armar el dataset de **Poster2Movie** ya tenemos repositorio:

🔗 <URL_DEL_REPO>

Cada equipo agrega películas **solo de sus géneros**, en su carpeta `contribuciones/equipoN/`. Todo está explicado en el README, pero en resumen:

1. **Saquen una API key de TMDB** (gratis): themoviedb.org → Settings → API.
2. **Clonen el repo** y corran `pip install -r requirements.txt`.
3. **Descarguen sus películas** con el script (ya filtra solas las que están en el dataset base o en otro equipo):
   `python scripts/descargar_tmdb.py --equipo equipoN --generos <ids de sus géneros> --max 300`
4. **Validen** con `python scripts/validar.py equipoN` y suban solo cuando diga OK.
5. **Suban** con `git add`, `commit`, `pull --rebase` y `push`, tocando únicamente su carpeta.

📌 Reglas importantes:
- Una película es del equipo de su **primer género** (el script lo respeta, no lo cambien).
- Sinopsis en inglés y carteles en JPG (el script ya lo hace así).
- No editen el CSV a mano ni suban el dataset de 43 GB.
- Prioricen **estrenos recientes** (`--desde 2023-01-01`) y **cine mexicano/latino** (`--pais MX` o `--idioma es`). Eso es lo que más le aporta al dataset base.

🎯 Meta: **500–1,000 películas por equipo** para el <FECHA>.

Cualquier duda me dicen.
