"""Estilo visual por género para el prompt del cartel (equipo 2).

ESTILOS_A: Rol 2 (8 géneros), versión "cinematográfica" (la mejor en su análisis).
ESTILOS_B: Rol 3 (9 géneros), versión 4 (iluminación avanzada).
Juntos cubren los 17 géneros de config.GENEROS. Cada estilo tiene ~10 palabras a propósito:
el codificador de texto de CLIP lee solo 77 tokens y un estilo largo se come la sinopsis.
"""

ESTILOS_A = {
    "Comedy": "cheerful pastel palette, natural soft lighting, balanced expressive framing",
    "Animation": "vivid complementary colors, cinematic glow, whimsical layered framing",
    "Family": "soft golden hues, warm ambient lighting, spacious balanced composition",
    "Fantasy": "mystical blue and gold hues, atmospheric backlighting, epic layered composition",
    "Adventure": "earthy green and gold palette, sunlight beams, expansive cinematic framing",
    "Action": "bold crimson and steel palette, intense directional lighting, dynamic perspective",
    "Science Fiction": "metallic blue and silver palette, atmospheric neon lighting, futuristic wide framing",
    "Mystery": "cold desaturated palette, moody directional lighting, deep shadow composition",
}

ESTILOS_B = {
    "Thriller": "Chiaroscuro lighting, deep violet undertones, sharp asymmetrical suspenseful framing",
    "Horror": "Sub-surface gloom, blinding emergency highlights, claustrophobic macro detail framing",
    "Crime": "Gritty neon reflections, deep shadow contrast, rugged urban street composition",
    "War": "Charred earth palette, harsh directional fire glow, frantic documentary-style framing",
    "Western": "Bleached desert sunlight, long dramatic shadows, grand heroic low-angle vistas",
    "Drama": "Soft diffused softbox lighting, rich earthy tones, empathetic character-driven composition",
    "Romance": "Ethereal backlight diffusion, warm rose-gold tones, intimate tender close-up framing",
    "History": "Rich oil painting palette, dramatic window chiaroscuro, stately historical composition",
    "Documentary": "Unfiltered ambient daylight, neutral color fidelity, crisp unvarnished realism framing",
}

ESTILOS = {**ESTILOS_A, **ESTILOS_B}
