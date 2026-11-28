"""Traducción al español de la encuesta (géneros y sinopsis de las 20 películas de la muestra de val).
Uso: from p2m import encuesta_es; encuesta_es.aplicar("/content/enc2/encuesta.html")
Guarda una copia encuesta_ingles.html al lado y deja el texto original como tooltip de cada sinopsis."""

import re, html, shutil
from pathlib import Path

GEN = {"Action":"Acción","Adventure":"Aventura","Animation":"Animación","Comedy":"Comedia","Crime":"Crimen",
       "Documentary":"Documental","Drama":"Drama","Family":"Familiar","Fantasy":"Fantasía","History":"Historia",
       "Horror":"Terror","Music":"Música","Mystery":"Misterio","Romance":"Romance","Science Fiction":"Ciencia ficción",
       "TV Movie":"Película de TV","Thriller":"Suspenso","War":"Bélica","Western":"Western"}

ES = [
"Un misterioso forastero llega a un pueblo en una motocicleta singular. Todo lo que lleva consigo es la Biblia y un deseo de justicia. La venganza del pasado estalla cuando Ryder repara una injusticia de su pasado y libera al pequeño pueblo de un malvado opresor.",
"Maxim Kammerer lucha por su amor y su libertad. Lidera un movimiento rebelde que desafía a los cinco codiciosos gobernantes. Uno de ellos, Strannik, sabe lo peligroso que puede ser Maxim y se propone detener al joven a cualquier precio. La última pelea cuerpo a cuerpo decidirá todo.",
"Gwynplaine, hijo de lord Clancharlie, lleva grabada en el rostro una sonrisa permanente, hecha por el rey como venganza por la traición de su padre. Un artista ambulante lo adopta y se convierte en un ídolo popular. Se enamora de la ciega Dea. El rey muere, y su malvado bufón intenta destruir o corromper a Gwynplaine.",
"Margaret Fish prepara una fiesta sorpresa para su esposo dentista, Bob. Mientras tanto, en la oficina, Bob atraviesa una crisis de la mediana edad mientras los insectos devoran lo que queda de sus plantas. Cuando Bob regresa a casa, a Margaret le cuesta muchísimo llevarlo al cuarto donde todos se esconden, hasta que él está a medio cambiarse y hablando de lo horribles que son todos sus mediocres…",
"El aclamado drama basado en la novela premiada de Dalene Matthee. “Fiela se Kind” (El hijo de Fiela) es la historia de un niño blanco abandonado que es criado por una familia de color. Pero la vida del niño cambia para siempre cuando unos funcionarios blancos del censo lo descubren viviendo al otro lado de las fronteras establecidas de la sociedad y se lo quitan a sus padres adoptivos. “Esta es la conmovedora historia de…",
"Cuando el hermano mayor al que idolatra (Tommy Nash) es sacado de la carretera por un despiadado narcotraficante (Darren Thomas) durante una carrera callejera nocturna conocida como Sepulveda Suicide, Rick Merchant (Jaz Martin) convierte su dolor en venganza al volante. Pero para ganar, tendrá que modificar su fiel Datsun 280Z de 1978, con la ayuda de una bonita mecánica (Hennely Jimenez), para…",
"Esta película contiene tres cortometrajes sobre el tema de la homosexualidad. En “A Friend of Dorothy”, una mujer se une a la Marina en los años 50 y descubre el lesbianismo. En “Mr. Roberts”, un maestro de un salón de clases de los años 70 lucha con el hecho de ocultar que es gay. Por último, en “Amos and Andy”, un padre lucha con su propia aceptación emocional de una boda actual entre…",
"Durval y su madre Carmita viven en la parte trasera de “Durval Discos”, una tienda de discos de vinilo de su propiedad en São Paulo. Llevan una vida aburrida y poco atractiva, hasta el día en que Durval contrata a una empleada, Celia, para ayudar a su madre con la casa. Sin embargo, al segundo día Celia se va y deja atrás a una niña de 5 años, Kiki, y una nota en la que promete volver…",
"En los turbios mercados negros y las chozas bombardeadas de Tokio después de la Segunda Guerra Mundial, un duro grupo de prostitutas sobrevive en una existencia de perro come perro, manteniendo amistades frágiles y un resto de orden en un mundo caótico. Pero cuando un ex soldado renegado irrumpe entre ellas, chocan los deseos y las lealtades, con resultados trágicos. Con Gate of Flesh, el director visionario Seijun Suzuki…",
"En el futuro, una monja guerrera en patines debe rescatar a una vidente que va a ser sacrificada por una banda de mutantes.",
"La gente come compulsivamente y sufre una sed extrema que los lleva a enloquecer por el agua. No pueden controlarse porque un parásito les ha lavado el cerebro.",
"De regreso a casa en avión rumbo a Nueva York, la adolescente Sarah se enferma gravemente. Se da cuenta demasiado tarde de que es más que un resfriado y, con sus padres todavía de vacaciones, se desploma en plena calle y muere poco después en un hospital, a causa de la peste, como descubren aterrados los forenses. Mientras intenta ocultarlo al público para evitar el pánico, el comisario de la epidemia…",
"A Stathis le encantan las motos, Rena quiere bailar todo el tiempo y Aleksis se vuelve loco por la música rock. Son tres hermanos alocados. Uno de ellos, Stathis, no parece capaz de conservar un empleo y, cuando pierde a su novia, su madre decide darle lo que más ama: ¡una motocicleta!",
"Cuando una mimada perrita de Hollywood es enviada al norte a vivir con un chucho de granja de lengua afilada, todas las correas se sueltan.",
"Katherine “Kathy” es feliz en su matrimonio con el empresario Greg Norris y vive en su hermosa casa con su hija Tara, de unos 18 años. De pronto, un encantador joven, a quien Tara vio antes merodeando por el jardín, aparece en su puerta y dice ser Jamie, el hijo que ella dio en adopción siendo madre adolescente por orden de sus padres, y la pareja se ve…",
"Cuando se descubre que un gran banco paga a un extorsionador corporativo, los medios y los fiscales empiezan a investigar y destapan un escándalo de dinero y favores que amenaza con sacudir toda la estructura de los negocios y del gobierno. Mientras los altos ejecutivos del banco siguen titubeando, un cuarteto de reformistas de mandos medios, liderado por el íntegro Kitano (Koji Yakusho), decide…",
"Un hombre rico contrata a dos guardaespaldas, uno exparacaidista (Qodirov) y el otro recién salido de prisión (Soipov), para proteger a sus dos hijas, Gulnoza (Ashurboeva) y Dilnoza (Eshonqulova). Aunque al principio las chicas se resisten a aceptar su destino, pronto se dan cuenta de la suerte que tienen de contar con estos dos jóvenes en sus vidas…",
"The Love Prophet and the Children of God es una mirada interna y fascinante a uno de los movimientos religiosos más enigmáticos del mundo y a su infame fundador, David Berg.",
"Un equipo de fuerzas especiales es enviado a acabar con un centro de venta de drogas, pero queda atrapado dentro tras ser traicionado y tendido una trampa.",
"Un joven grafitero rebelde, que ataca las casas de la élite adinerada, descubre un secreto impactante que lo lleva a un viaje que pone en peligro a él y a los más cercanos a él.",
]

def aplicar(ruta):
    ruta = Path(ruta); texto = ruta.read_text(encoding="utf-8")
    copia = ruta.with_name("encuesta_ingles.html")
    if not copia.exists():
        shutil.copy(ruta, copia)
    if 'title="' in texto.split('class="sinopsis"')[1][:20]:
        raise SystemExit("Este archivo ya está traducido; no se vuelve a aplicar.")
    n_sin = len(re.findall(r'<p class="sinopsis">', texto))
    assert n_sin == len(ES) == 20, f"Se esperaban 20 sinopsis y hay {n_sin} en el HTML"
    def gen(m):
        g = [GEN.get(x.strip(), x.strip()) for x in html.unescape(m.group(1)).split(",")]
        return "<small>(" + html.escape(", ".join(g)) + ")</small>"
    texto, n1 = re.subn(r"<small>\((.*?)\)</small>", gen, texto)
    it = iter(ES)
    def sin(m):
        en = html.unescape(m.group(1))
        return f'<p class="sinopsis" title="{html.escape(en, quote=True)}">{html.escape(next(it))}</p>'
    texto, n2 = re.subn(r'<p class="sinopsis">(.*?)</p>', sin, texto, flags=re.S)
    ruta.write_text(texto, encoding="utf-8")
    print(f"Géneros traducidos: {n1} | sinopsis traducidas: {n2}")

