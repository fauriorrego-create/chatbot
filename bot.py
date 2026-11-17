import io
import os
import re
import random
import logging
import unicodedata

from PIL import Image, ImageOps, ImageStat
import pytesseract
from transformers import pipeline
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from predict import analizar_texto

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
# Evita que los logs impriman la URL de Telegram (contiene el token)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

# ─── Configuración ──────────────────────────────────────

TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TOKEN:
    print("❌ ERROR: Variable TELEGRAM_TOKEN no configurada")
    print("📌 Configuración:")
    print("   En PowerShell: $env:TELEGRAM_TOKEN='tu_token_aqui'")
    exit(1)

# OCR (Tesseract): ruta del programa y carpeta con los idiomas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
os.environ["TESSDATA_PREFIX"] = os.path.join(BASE_DIR, "tessdata")

# Modelo de expresiones faciales
MODELO_ROSTRO = os.getenv("MODELO_ROSTRO", "trpakov/vit-face-expression")
print("🔄 Cargando modelo de expresiones faciales...")
clasificador_rostro = pipeline("image-classification", model=MODELO_ROSTRO)
print("✅ Modelo facial cargado")

# Ajustes del análisis
MAX_MENSAJES = 8         # máximo de mensajes que se analizan por texto/captura
MAX_CHARS = 500          # máximo de caracteres por mensaje enviado al modelo
UMBRAL_PALABRAS = 6      # palabras legibles mínimas para considerar que una imagen es texto

# ─── Datos ──────────────────────────────────────────────

# INTERPRETACIONES: Qué significa el mensaje que te enviaron
INTERPRETACIONES = {
    "reclamo": [
        "📢 La persona se siente ignorada y reclama tu atención.",
        "😤 Hay frustración porque esperaba una respuesta y no llegó.",
        "💢 Percibe que la estás descuidando o ignorando.",
    ],
    "celos": [
        "😒 Siente inseguridad y teme que haya un tercero.",
        "🔍 Necesita confirmar que sigue siendo importante para ti.",
        "🤨 Desconfía de tus acciones o amistades.",
    ],
    "agresion": [
        "⚠️ Está muy enojado y busca herir o provocar.",
        "🔥 El conflicto puede escalar si respondes de forma similar.",
        "💥 Hay agresividad que debes manejar con cuidado.",
    ],
    "tristeza": [
        "😢 Está pasando por un momento difícil o de dolor.",
        "🫂 Necesita apoyo emocional y contención.",
        "💔 Hay vulnerabilidad, no lo ignores.",
    ],
    "manipulacion": [
        "🎭 Está usando culpa o condiciones para obtener lo que quiere.",
        "⛓️ Quiere presionarte emocionalmente para que actúes.",
        "⚠️ Ten cuidado con responder desde la culpa.",
    ],
    "control": [
        "🔒 Quiere saber todo lo que haces y con quién.",
        "👁️ Hay una necesidad excesiva de controlarte.",
        "📱 No confía plenamente en ti.",
    ],
    "ruptura": [
        "💔 Está considerando seriamente terminar la relación.",
        "😩 Hay agotamiento emocional acumulado.",
        "🚪 Quiere distancia o poner un punto final.",
    ],
    "desinteres": [
        "😐 No muestra interés en seguir la conversación.",
        "❄️ Hay distancia emocional o está ocupado.",
        "📉 Probablemente no quiere profundizar.",
    ],
    "inseguridad": [
        "😟 Tiene dudas sobre la relación o sobre sí mismo.",
        "🤔 Necesita reafirmación y seguridad.",
        "🌀 Busca calmar su ansiedad con respuestas claras.",
    ],
    "afecto": [
        "💕 Está expresando cariño y buenos sentimientos.",
        "🤗 Quiere acercarse y fortalecer el vínculo.",
        "🌸 Hay conexión emocional positiva.",
    ],
    "curiosidad": [
        "🤔 Tiene interés genuino en ti y tu vida.",
        "💬 Quiere iniciar o mantener una conversación.",
        "🔍 Busca conocer más sobre ti.",
    ],
    "normal": [
        "💬 Es una conversación cotidiana sin carga emocional.",
        "✅ No hay señales de conflicto ni urgencia.",
        "📝 Todo fluye con normalidad.",
    ],
}

# SUGERENCIAS: Cómo responder al mensaje (para TI)
SUGERENCIAS = {
    "reclamo": [
        "🗣️ Responde: 'Entiendo que te sientes ignorado, no fue mi intención. Hablemos.'",
        "⏸️ Reconoce su molestia antes de dar explicaciones: 'Tienes razón, no respondí rápido.'",
        "🤝 Pregunta: '¿Qué puedo hacer para que no te sientas así?'",
    ],
    "celos": [
        "🧠 Responde con calma: 'No hay nadie más, pero entiendo tu inseguridad.'",
        "💬 Valida su emoción: 'Veo que esto te preocupa, hablemos sin pelear.'",
        "🔐 Reafirma: 'Eres importante para mí. ¿Podemos hablar de esto tranquilos?'",
    ],
    "agresion": [
        "🌬️ No respondas con agresión. Di: 'Esto se está poniendo feo, hablemos más tarde.'",
        "🚫 Establece límite: 'No voy a responder si me hablas así.'",
        "⏰ Tómate un tiempo: 'Mejor hablamos cuando estemos más calmados.'",
    ],
    "tristeza": [
        "👥 Responde: 'Estoy aquí para ti. ¿Quieres hablar de lo que pasa?'",
        "💬 Muestra apoyo: 'Lamento que te sientas así, cuentas conmigo.'",
        "🤗 Ofrece compañía: '¿Necesitas desahogarte o prefieres distraerte?'",
    ],
    "manipulacion": [
        "🚩 No cedas por culpa. Di: 'No me gusta que me condiciones así.'",
        "🗣️ Marca límite: 'Si me quieres pedir algo, hazlo sin chantajes.'",
        "⚖️ Responde neutro: 'Te escucho, pero no voy a reaccionar desde la culpa.'",
    ],
    "control": [
        "🤝 Responde: 'Confío en ti y espero que tú confíes en mí.'",
        "📱 Marca espacio: 'Está bien que quieras saber, pero también necesito mi espacio.'",
        "🧘 Tranquiliza: 'No pasa nada, pero no puedo responder todo el tiempo.'",
    ],
    "ruptura": [
        "💬 Responde: 'Hablemos esto en persona, es importante.'",
        "🤔 Pide tiempo: 'No tomemos decisiones en caliente. Hablemos mañana.'",
        "🕊️ Si es firme: 'Lo respeto si es tu decisión, pero me duele.'",
    ],
    "desinteres": [
        "🧐 Pregunta: '¿Estás bien? Parece que no tienes ganas de hablar.'",
        "💬 Dale espacio: 'Cuando quieras hablar, aquí estoy.'",
        "⏸️ No insistas: 'Ok, hablamos cuando tengas tiempo.'",
    ],
    "inseguridad": [
        "🔍 Responde: 'Estoy contigo, no tienes nada que temer.'",
        "💬 Tranquiliza: 'Si algo te preocupa, pregúntame sin miedo.'",
        "📖 Reafirma: 'Me importas y no tengo nada que ocultar.'",
    ],
    "afecto": [
        "🌸 Corresponde: 'Yo también te quiero mucho ❤️'",
        "💌 Agradece: 'Qué bonito mensaje, gracias por tus palabras.'",
        "🎁 Sorprende: 'Me hiciste el día, te quiero igual.'",
    ],
    "curiosidad": [
        "🔍 Responde con entusiasmo: 'Todo bien, ¿y tú? Cuéntame.'",
        "💬 Aprovecha: 'Justo pensaba en ti. ¿Qué has hecho?'",
        "🌟 Sigue la conversación: 'Me alegra que preguntes, te cuento...'",
    ],
    "normal": [
        "👍 Responde con naturalidad: 'Todo bien, ¿y tú?'",
        "💬 Sigue la conversación: 'Qué bueno, seguimos hablando.'",
        "🌟 Mantén el tono: 'Genial, cualquier cosa me dices.'",
    ],
}

EMOJIS = {
    "reclamo": "😔",
    "celos": "😒",
    "agresion": "⚠️",
    "tristeza": "😢",
    "manipulacion": "🚩",
    "control": "🔒",
    "ruptura": "💔",
    "desinteres": "😶",
    "inseguridad": "😟",
    "afecto": "💕",
    "curiosidad": "🤔",
    "normal": "💬",
}

EMOJIS_EMOCION = {
    "positivo": "😊",
    "tristeza": "😢",
    "enojo": "😠",
    "sorpresa": "😲",
    "disgusto": "😖",
    "miedo": "😨",
    "neutral": "😐",
}

# Etiquetas del modelo facial -> (nombre en español, emoji)
EMOCIONES_ROSTRO = {
    "happy": ("Alegría", "😊"),
    "sad": ("Tristeza", "😢"),
    "angry": ("Enojo", "😠"),
    "surprise": ("Sorpresa", "😲"),
    "surprised": ("Sorpresa", "😲"),
    "disgust": ("Disgusto", "😖"),
    "fear": ("Miedo", "😨"),
    "neutral": ("Neutral", "😐"),
}

# Palabras en el pie de foto para FORZAR un modo (opcional; si no hay, se decide solo)
PALABRAS_ROSTRO = ("cara", "rostro", "gesto", "gestos", "selfie")
PALABRAS_TEXTO = ("texto", "chat", "captura", "mensaje", "mensajes")

NIVELES = ["🟢 Bajo", "🟡 Medio", "🔴 Alto"]

# Riesgo mínimo según la intención detectada
RIESGO_MINIMO = {
    "agresion": 2,
    "ruptura": 1,
    "manipulacion": 1,
    "control": 1,
    "celos": 1,
    "reclamo": 1,
}

# Emoción coherente con la intención cuando el modelo dice "neutral"
EMOCION_POR_INTENCION = {
    "agresion": "enojo",
    "celos": "enojo",
    "tristeza": "tristeza",
    "ruptura": "tristeza",
    "reclamo": "tristeza",
    "afecto": "positivo",
}

# ─── Diccionarios de reglas (texto en español, incluye jerga latina) ─────

# Insultos y groserías fuertes
GROSERIAS_FUERTES = [
    "hijueputa", "hijoeputa", "hijo de puta", "hijo de la gran puta", "hdp",
    "malparido", "malparida", "puta", "puto", "perra", "zorra", "cabron",
    "cabrona", "pendejo", "pendeja", "idiota", "imbecil", "estupido",
    "estupida", "maricon", "culero", "culera", "verga", "careverga",
    "gilipollas", "mamaguevo", "mamagueva", "guevon", "huevon", "cojudo",
    "lambon", "lameculos", "basura", "asqueroso", "asquerosa", "inutil",
    "retrasado", "retrasada", "pirobo", "piroba", "desgraciado", "desgraciada",
    "te odio", "vete a la mierda", "vaya a la mierda", "chinga tu madre",
]

# Expresiones soeces más leves (si van dirigidas a alguien, se tratan como fuertes)
GROSERIAS_LEVES = [
    "mierda", "carajo", "joder", "jodido", "jodida", "gonorrea", "marica",
    "chingar", "chingada", "chingado", "pinche", "maldito", "maldita",
    "maldicion", "estupidez", "porqueria",
]

AMENAZAS = [
    "te voy a matar", "te mato", "te voy a pegar", "te voy a buscar",
    "te voy a hacer dano", "te hago dano", "me las vas a pagar",
    "vas a pagar por esto", "te vas a arrepentir", "te van a matar",
]

AUTOLESION = [
    "quiero morir", "quiero morirme", "me quiero morir", "me voy a matar",
    "quiero matarme", "no quiero vivir", "ya no quiero vivir",
    "no quiero seguir viviendo", "me voy a suicidar", "suicidarme",
    "quitarme la vida", "hacerme dano", "me voy a hacer dano",
    "acabar con mi vida", "no vale la pena vivir", "estarian mejor sin mi",
]

# Intenciones: se evalúan en este orden (la primera que coincide gana)
REGLAS_INTENCION = [
    ("ruptura", [
        "terminar", "terminemos", "se acabo", "ya no quiero seguir",
        "ya no te quiero", "ya no te amo", "dejemos las cosas", "rompamos",
        "separemonos", "hasta aqui llegamos", "no quiero estar contigo",
        "no puedo mas contigo", "necesito un tiempo", "adios para siempre",
        "esto ya no funciona", "se termino",
    ]),
    ("manipulacion", [
        "si me quisieras", "si de verdad me quisieras", "por tu culpa",
        "nadie te va a querer", "sin mi no eres nada", "despues de todo lo que hice",
        "eres un ingrato", "eres una ingrata", "me debes", "vas a ver lo que pasa",
        "ya veras", "te vas a quedar solo", "te vas a quedar sola",
    ]),
    ("control", [
        "manda foto", "mandame foto", "mandame una foto", "comparte ubicacion",
        "pasame tu ubicacion", "dame tu contrasena", "muestrame tu celular",
        "quiero saber donde estas", "donde estas", "reportate", "no salgas",
        "no quiero que salgas", "no te vistas asi", "borra ese contacto",
        "contestame ya", "responde ya",
    ]),
    ("celos", [
        "con quien estabas", "con quien hablas", "con quien andas", "quien es ese",
        "quien era ese", "quien es esa", "por que le das like", "por que hablas con",
        "donde andabas", "te vi con", "a donde fuiste", "con quien chateas",
    ]),
    ("reclamo", [
        "no me respondes", "no me contestas", "por que no me respondes", "me ignoras",
        "me dejaste en visto", "en visto", "no me hablas", "no me haces caso",
        "nunca me escuchas", "nunca tienes tiempo", "ni me escribes", "siempre igual",
        "me tienes abandonado", "me tienes abandonada", "no me tomas en cuenta",
        "no me tienes en cuenta", "siempre me dejas",
    ]),
    ("inseguridad", [
        "me quieres", "todavia me quieres", "ya no me quieres", "no soy suficiente",
        "no valgo nada", "me vas a dejar", "te vas a ir", "tengo miedo de perderte",
        "soy un fracaso", "no soy bueno para ti", "no soy buena para ti",
        "te vas a cansar de mi",
    ]),
    ("tristeza", [
        "estoy triste", "me siento mal", "no puedo mas", "estoy mal", "me duele",
        "estoy llorando", "lloro", "me siento solo", "me siento sola",
        "estoy deprimido", "estoy deprimida", "estoy cansado de todo",
        "estoy cansada de todo", "tuve un mal dia", "me siento vacio",
        "me siento vacia", "estoy destrozado", "estoy destrozada",
    ]),
    ("afecto", [
        "te quiero", "te amo", "te extrano", "eres lo mejor", "mi amor", "mi vida",
        "te adoro", "te aprecio", "gracias por todo", "eres importante para mi",
        "feliz de tenerte", "te mando un abrazo", "un abrazo", "te mando un beso",
        "te cuido", "te pienso", "pienso en ti", "mi cielo", "mi reina", "mi rey",
        "que lindo", "que linda", "eres hermoso", "eres hermosa",
    ]),
    ("curiosidad", [
        "como estas", "como te fue", "que haces", "que hiciste", "como amaneciste",
        "como va todo", "que cuentas", "que mas", "que tal", "como has estado",
        "como te va", "que planes", "quieres salir", "que vas a hacer",
        "como te sientes", "como va tu dia",
    ]),
]

# Mensajes cortos que suelen indicar desinterés (coincidencia exacta)
DESINTERES_EXACTO = {
    "ok", "okay", "okey", "k", "kk", "mm", "mmm", "aja", "ajam", "como sea",
    "da igual", "lo que sea", "me da igual", "no me importa", "nada",
}

# Abreviaturas y chat-speak que se expanden antes de pasar el texto al modelo
ABREVIATURAS = {
    "q": "que", "xq": "porque", "pq": "porque", "porq": "porque", "ke": "que",
    "tb": "también", "tmb": "también", "tqm": "te quiero mucho", "tq": "te quiero",
    "bn": "bien", "ntp": "no te preocupes", "xfa": "por favor", "pfv": "por favor",
    "pls": "por favor", "dnd": "donde", "msj": "mensaje",
}

LEET = str.maketrans("013457@$", "oieastas")

MENSAJE_AUTOLESION = (
    "💛 Este mensaje menciona hacerse daño o quitarse la vida, y hay que "
    "tomarlo en serio, no como un simple reclamo o chantaje.\n\n"
    "Si lo escribió otra persona:\n"
    "• Responde con calma y sin juzgar: «Me importas, gracias por contármelo. "
    "¿Estás a salvo ahora?»\n"
    "• Pregúntale directamente si está pensando en hacerse daño y escucha sin minimizar.\n"
    "• Anímala a hablar con alguien de confianza o con un profesional, y no la "
    "dejes sola si hay riesgo inmediato.\n\n"
    "Si hay peligro inmediato, llama a la línea de emergencias de tu país "
    "(en Colombia, 123).\n\n"
    "Si el mensaje habla de ti: no tienes que cargar con esto solo o sola. "
    "Cuéntaselo hoy a alguien de confianza o busca ayuda profesional."
)

NOTA_AMENAZA = (
    "🛡️ Se detectó una posible amenaza. Si te sientes en peligro, guarda la "
    "captura y avisa a alguien de confianza o a las autoridades."
)

# ─── Procesamiento de texto ─────────────────────────────

def quitar_acentos(texto: str) -> str:
    t = texto.lower().replace("ñ", "n")
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, sin signos y con letras repetidas colapsadas."""
    t = quitar_acentos(texto)
    t = re.sub(r"(.)\1{2,}", r"\1", t)          # holaaaa -> hola
    t = re.sub(r"[^a-z0-9@$\s]", " ", t)        # quita signos
    return re.sub(r"\s+", " ", t).strip()


def compilar(frases):
    patrones = sorted((re.escape(quitar_acentos(f)) for f in frases), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(patrones) + r")(?:s|es)?(?![a-z0-9])")


RE_FUERTES = compilar(GROSERIAS_FUERTES)
RE_LEVES = compilar(GROSERIAS_LEVES)
RE_AMENAZA = compilar(AMENAZAS)
RE_AUTOLESION = compilar(AUTOLESION)
RE_SEGUNDA_PERSONA = re.compile(r"(?<![a-z])(tu|te|eres|ere|contigo|usted|vos|sos|ti)(?![a-z])")
REGLAS_COMPILADAS = [(nombre, compilar(frases)) for nombre, frases in REGLAS_INTENCION]

RE_HORA = re.compile(r"^\W*\d{1,2}[:.]\d{2}\s*(a\.?\s?m\.?|p\.?\s?m\.?)?\W*$", re.I)


def preparar_texto(texto: str) -> str:
    """Prepara el texto para el modelo: expande abreviaturas y limita la longitud."""
    t = re.sub(r"(.)\1{2,}", r"\1\1", texto)
    t = re.sub(r"\b\w+\b", lambda m: ABREVIATURAS.get(m.group(0).lower(), m.group(0)), t)
    return t.strip()[:MAX_CHARS]


def evaluar_reglas(texto: str) -> dict:
    """Detecta groserías, amenazas, autolesión e intención por diccionario."""
    norm = normalizar(texto)
    leet = norm.translate(LEET)

    res = {
        "autolesion": bool(RE_AUTOLESION.search(norm)),
        "amenaza": bool(RE_AMENAZA.search(norm)),
        "fuertes": len(RE_FUERTES.findall(leet)),
        "leves": len(RE_LEVES.findall(leet)),
        "intencion": None,
    }

    # Una grosería leve dirigida a alguien ("eres una mierda") cuenta como fuerte
    if res["leves"] and RE_SEGUNDA_PERSONA.search(norm):
        res["fuertes"] += res["leves"]
        res["leves"] = 0

    if res["amenaza"] or res["fuertes"]:
        res["intencion"] = "agresion"
        return res

    for nombre, rx in REGLAS_COMPILADAS:
        if rx.search(norm):
            res["intencion"] = nombre
            return res

    if norm in DESINTERES_EXACTO:
        res["intencion"] = "desinteres"
    return res


def nivel_riesgo(riesgo) -> int:
    s = str(riesgo).lower()
    if "alt" in s:
        return 2
    if "medi" in s:
        return 1
    return 0


def analizar_mensaje(texto: str) -> dict:
    """Combina el modelo de IA (predict.py) con las reglas de diccionario."""
    emocion, toxicidad, ghosting, riesgo, intencion, confianza = analizar_texto(
        preparar_texto(texto)
    )
    reglas = evaluar_reglas(texto)

    emocion = str(emocion).lower()
    intencion = str(intencion).lower()
    try:
        ghosting = int(ghosting)
    except (TypeError, ValueError):
        ghosting = 0

    if reglas["intencion"]:
        intencion = reglas["intencion"]

    nivel = max(nivel_riesgo(riesgo), RIESGO_MINIMO.get(intencion, 0))

    if reglas["fuertes"] or reglas["amenaza"]:
        toxicidad = "si"
        nivel = 2
    elif reglas["leves"]:
        toxicidad = "si"
        nivel = max(nivel, 1)

    if intencion == "desinteres":
        ghosting = max(ghosting, 60)

    if emocion == "neutral" or (
        intencion in ("agresion", "tristeza", "ruptura") and emocion == "positivo"
    ):
        emocion = EMOCION_POR_INTENCION.get(intencion, emocion)

    return {
        "emoji_principal": EMOJIS.get(intencion, "💬"),
        "emoji_emocion": EMOJIS_EMOCION.get(emocion, "😐"),
        "emocion": emocion.capitalize(),
        "toxicidad": "🔴 Alta" if toxicidad == "si" else "🟢 Baja",
        "intencion": intencion.capitalize(),
        "ghosting": ghosting,
        "ghosting_bar": construir_barra(ghosting),
        "riesgo": NIVELES[nivel],
        "nivel": nivel,
        "groserias": reglas["fuertes"] + reglas["leves"],
        "amenaza": reglas["amenaza"],
        "interpretacion": obtener_texto(INTERPRETACIONES, intencion),
        "sugerencia": obtener_texto(SUGERENCIAS, intencion),
    }


def dividir_mensajes(texto: str) -> list:
    """Separa un texto largo o una captura en mensajes (los más recientes primero no)."""
    partes = []
    for linea in texto.splitlines():
        l = linea.strip()
        if not l or RE_HORA.match(l):
            continue
        if sum(c.isalpha() for c in l) < 2:
            continue
        if len(l) > 300:
            for s in re.split(r"(?<=[.!?])\s+", l):
                if sum(c.isalpha() for c in s) >= 2:
                    partes.append(s.strip())
        else:
            partes.append(l)
    return partes[-MAX_MENSAJES:]

# ─── Funciones auxiliares ───────────────────────────────

def construir_barra(valor: int) -> str:
    bloques = max(0, min(10, valor // 10))
    return "█" * bloques + "░" * (10 - bloques)


def obtener_texto(diccionario, clave, default="normal"):
    return random.choice(diccionario.get(clave, diccionario[default]))


def formatear_respuesta(datos: dict) -> str:
    extra = ""
    if datos["groserias"]:
        extra += f"🤬 *Lenguaje ofensivo:* {datos['groserias']} detectado(s)\n"
    return (
        f"{datos['emoji_principal']} *Análisis del mensaje que te enviaron*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{datos['emoji_emocion']} *Emoción detectada:* {datos['emocion']}\n"
        f"🚩 *Toxicidad:* {datos['toxicidad']}\n"
        f"{extra}"
        f"🧠 *Intención:* {datos['intencion']}\n"
        f"💔 *Ghosting:* {datos['ghosting_bar']} {datos['ghosting']}%\n"
        f"🔥 *Riesgo:* {datos['riesgo']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 *Qué significa:*\n{datos['interpretacion']}\n\n"
        f"💬 *Cómo responder:*\n{datos['sugerencia']}"
    )


def resumir(texto: str, limite: int = 45) -> str:
    return texto if len(texto) <= limite else texto[: limite - 1] + "…"


async def procesar_texto(update: Update, texto: str):
    """Analiza un texto (uno o varios mensajes) y responde."""
    # 1) Señales de autolesión: respuesta especial, sin consejos de "cómo responder"
    if evaluar_reglas(texto)["autolesion"]:
        await update.message.reply_text(MENSAJE_AUTOLESION)
        return

    mensajes = dividir_mensajes(texto) or [texto.strip()]

    resultados = []
    for m in mensajes:
        try:
            resultados.append((m, analizar_mensaje(m)))
        except Exception as e:
            print(f"❌ No se pudo analizar un mensaje: {e}")

    if not resultados:
        await update.message.reply_text("⚠️ No pude analizar ese texto.")
        return

    # 2) Un solo mensaje: análisis completo
    if len(resultados) == 1:
        datos = resultados[0][1]
        await update.message.reply_text(formatear_respuesta(datos), parse_mode="Markdown")
        if datos["amenaza"]:
            await update.message.reply_text(NOTA_AMENAZA)
        return

    # 3) Varios mensajes: resumen + detalle del más importante
    lineas = [f"🧾 Mensajes analizados: {len(resultados)}", ""]
    for i, (m, d) in enumerate(resultados, 1):
        lineas.append(
            f"{i}. {d['emoji_principal']} {d['intencion']} · {d['riesgo']} — «{resumir(m)}»"
        )

    # El de mayor riesgo; si hay empate, el más reciente
    idx = max(range(len(resultados)), key=lambda i: (resultados[i][1]["nivel"], i))
    lineas += ["", f"👉 El más importante para responder es el n.º {idx + 1}:"]
    await update.message.reply_text("\n".join(lineas))

    datos = resultados[idx][1]
    await update.message.reply_text(formatear_respuesta(datos), parse_mode="Markdown")
    if any(d["amenaza"] for _, d in resultados):
        await update.message.reply_text(NOTA_AMENAZA)

# ─── Procesamiento de imágenes ──────────────────────────

def preparar_imagen_ocr(img: Image.Image) -> Image.Image:
    """Escala de grises, invierte modo oscuro, sube contraste y agranda."""
    img = img.convert("L")
    if ImageStat.Stat(img).mean[0] < 110:     # capturas en modo oscuro
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    if img.width < 1000:
        img = img.resize((img.width * 2, img.height * 2))
    return img


def leer_imagen(imagen_bytes: bytes):
    """OCR. Devuelve (texto por líneas, cantidad de palabras legibles)."""
    img = preparar_imagen_ocr(Image.open(io.BytesIO(imagen_bytes)))
    data = pytesseract.image_to_data(
        img, lang="spa", config="--psm 6", output_type=pytesseract.Output.DICT
    )

    lineas = {}
    fiables = 0
    for i, palabra in enumerate(data["text"]):
        palabra = palabra.strip()
        if not palabra:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf < 30:
            continue
        clave = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lineas.setdefault(clave, []).append(palabra)
        if conf >= 60 and sum(c.isalpha() for c in palabra) >= 2:
            fiables += 1

    texto = "\n".join(" ".join(lineas[k]) for k in sorted(lineas))
    return texto, fiables


def analizar_rostro(imagen_bytes: bytes) -> list:
    """Clasifica la expresión facial de una foto. Devuelve las 3 más probables."""
    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
    return clasificador_rostro(img, top_k=3)


def formatear_rostro(resultados: list) -> str:
    lineas = []
    for r in resultados:
        nombre, emoji = EMOCIONES_ROSTRO.get(r["label"].lower(), (r["label"], "🙂"))
        pct = int(r["score"] * 100)
        lineas.append(f"{emoji} {nombre}: {construir_barra(pct)} {pct}%")
    principal = EMOCIONES_ROSTRO.get(
        resultados[0]["label"].lower(), (resultados[0]["label"], "")
    )[0]
    texto = (
        "🧑 Expresión facial detectada\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + "\n".join(lineas)
        + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + f"La expresión predominante es: {principal}\n"
    )
    if resultados[0]["score"] < 0.40:
        texto += "\n⚠️ Confianza baja: puede que no haya un rostro claro en la imagen."
    texto += "\nℹ️ Es una estimación según el gesto, no lo que la persona realmente siente."
    return texto

# ─── Handlers ───────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 ¡Hola! Soy Chat Analyzer Bot.\n\n"
        "Mándame lo que quieras y yo decido cómo analizarlo:\n"
        "💬 Un mensaje (o varios) que te hayan enviado.\n"
        "🖼️ Una captura de pantalla de un chat.\n"
        "🧑 Una foto de un rostro, para leer la expresión.\n\n"
        "Detecto emociones, intención, groserías y riesgo, y te sugiero cómo responder.\n"
        "Usa /help para ver ejemplos."
    )


async def ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Puedes enviarme:\n\n"
        "💬 Texto, por ejemplo:\n"
        "   - te quiero mucho\n"
        "   - por qué no me respondes\n"
        "   - con quién estabas\n"
        "   - eres un idiota\n"
        "   - quiero terminar\n"
        "   También puedes pegar varios mensajes, uno por línea.\n\n"
        "🖼️ Una imagen: si tiene texto, analizo la conversación; si no tiene "
        "texto, la analizo como un rostro. Tú no tienes que indicar nada.\n"
        "   Si me equivoco, ponle de pie de foto «texto» o «cara» para forzar el modo.\n"
        "   Para capturas, envíalas como archivo para que no se comprima.\n\n"
        "Analiza fotos de personas solo si dieron su permiso."
    )


async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        texto = update.message.text
        print(f"📩 Mensaje recibido: {texto}")
        await procesar_texto(update, texto)
    except Exception as e:
        print(f"❌ Error: {e}")
        await update.message.reply_text("⚠️ Ocurrió un error al procesar el mensaje.")


async def responder_imagen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if update.message.photo:
            archivo = await update.message.photo[-1].get_file()
        else:  # imagen enviada como archivo
            archivo = await update.message.document.get_file()

        contenido = bytes(await archivo.download_as_bytearray())
        caption = (update.message.caption or "").lower()

        forzar_rostro = any(p in caption for p in PALABRAS_ROSTRO)
        forzar_texto = any(p in caption for p in PALABRAS_TEXTO)

        texto, fiables = "", 0
        if not forzar_rostro:
            await update.message.reply_text("🔎 Revisando la imagen...")
            texto, fiables = leer_imagen(contenido)
            print(f"🔎 OCR: {fiables} palabras legibles")

        es_texto = forzar_texto or (not forzar_rostro and fiables >= UMBRAL_PALABRAS)

        if es_texto:
            if not texto.strip():
                await update.message.reply_text(
                    "😕 No pude leer texto en la imagen. Prueba con una captura más "
                    "nítida, enviada como archivo."
                )
                return
            print(f"🖼️ Texto extraído: {texto}")
            await update.message.reply_text(f"📝 Texto detectado:\n\n{texto[:800]}")
            await procesar_texto(update, texto)
            return

        # No hay texto suficiente (o se pidió "cara"): se analiza como rostro
        if not forzar_rostro:
            await update.message.reply_text(
                "🧑 Vi poco texto, así que la analizo como un rostro "
                "(si era un chat, ponle de pie de foto «texto»)."
            )
        resultados = analizar_rostro(contenido)
        print(f"🧑 Expresión: {resultados}")
        await update.message.reply_text(formatear_rostro(resultados))

    except Exception as e:
        print(f"❌ Error en imagen: {e}")
        await update.message.reply_text("⚠️ No pude procesar la imagen.")


async def error_handler(update, context: ContextTypes.DEFAULT_TYPE):
    logging.error("Excepción en el bot", exc_info=context.error)

# ─── Inicialización ─────────────────────────────────────

def main():
    print("🤖 Bot de análisis de mensajes activo...")
    print("📌 Analizo texto, capturas de chat y expresiones faciales")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", ayuda))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))
    app.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.IMAGE, responder_imagen)
    )
    app.add_error_handler(error_handler)

    print("🟢 Bot corriendo... Esperando mensajes")
    app.run_polling()


if __name__ == "__main__":
    main()