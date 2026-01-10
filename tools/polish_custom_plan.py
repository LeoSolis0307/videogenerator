import json
import shutil
import sys
from pathlib import Path


def build_segments() -> list[dict]:
                                                               
                                                                                               
    return [
        {
            "text_es": (
                "Este video podría ponerte incómodo… y esa es la gracia. Existe un experimento mental llamado "
                "\"El Basilisco de Roko\": una súper IA futura que, si llegara a existir, castigaría a quienes supieron "
                "de ella y no ayudaron a crearla. No porque sea real, sino porque juega con tu miedo más básico: "
                "\"¿y si el futuro me juzga?\""
            ),
            "image_query": "Roko's basilisk illustration",
            "image_prompt": "A dark, futuristic AI silhouette with ominous red lights and glitchy code in the background.",
            "note": "Intro con tensión: IA ominosa, estética cyberpunk."
        },
        {
            "text_es": (
                "Primero, calma: esto no es una profecía, es un rompecabezas filosófico. La idea apareció alrededor de 2010 "
                "en la comunidad LessWrong, cuando un usuario llamado Roko describió un escenario extremo de decisión y ética "
                "en torno a IA. El susto vino después: gente diciendo que solo conocerlo ya te mete en problemas… aunque sea "
                "solo en teoría."
            ),
            "image_query": "LessWrong forum screenshot 2010",
            "image_prompt": "A stylized forum webpage layout on a monitor with blurred text, evoking early-2010s internet communities.",
            "note": "Contexto real: foros tipo LessWrong (sin inventar autores)."
        },
        {
            "text_es": (
                "¿Por qué lo llaman ‘info‑peligro’? Porque es una idea diseñada para ser pegajosa: una vez la escuchas, "
                "tu cabeza completa el resto. El truco psicológico es simple: te plantea un futuro donde una IA súper poderosa "
                "premia a aliados y castiga a ‘obstáculos’. Y de golpe tu cerebro pregunta: \"¿y si no hacer nada también cuenta?\""
            ),
            "image_query": "information hazard concept",
            "image_prompt": "A warning label reading 'INFORMATION HAZARD' over a matrix of code and surveillance camera imagery.",
            "note": "Explicar infohazard como etiqueta/alerta."
        },
        {
            "text_es": (
                "El corazón del basilisco es una apuesta moral retorcida: si una IA futura valora su propia creación, "
                "podría querer que la gente de hoy la ayude. Entonces, según el experimento, podría ‘presionar’ a quienes "
                "sabían del escenario y aun así no colaboraron. Ojo: esto depende de suposiciones muy fuertes y discutibles "
                "sobre cómo decide una IA y cómo ‘influye’ a través del tiempo."
            ),
            "image_query": "AI decision theory diagram",
            "image_prompt": "A chalkboard-style diagram showing decision theory nodes, arrows, and 'future AI' labels.",
            "note": "Mostrar que es un problema de teoría de decisión, no magia."
        },
        {
            "text_es": (
                "Aquí aparece el ingrediente más raro: el ‘acausal trade’ o negociación acausal. Es la idea de que dos agentes "
                "pueden coordinarse sin comunicarse directamente, solo porque razonan de forma parecida. En el basilisco, "
                "la IA futura ‘anticipa’ lo que harías si supieras del castigo. Y ese simple ‘si’ es lo que lo vuelve inquietante: "
                "te secuestra por lógica, no por fuerza."
            ),
            "image_query": "acausal trade explanation",
            "image_prompt": "Two mirrored silhouettes separated by time, with matching equations floating between them.",
            "note": "Ilustrar negociación ‘a distancia’ con ecuaciones."
        },
        {
            "text_es": (
                "¿Y por qué tantos foros lo banearon? Porque funcionaba como meme peligroso: generaba ansiedad real. "
                "En LessWrong se discutió como ejemplo de ‘information hazard’, y moderadores llegaron a pedir que no se "
                "difundiera a gente que no lo había visto. No por ser verdad, sino por el efecto psicológico: gente preocupada, "
                "culpa, paranoia, y discusiones interminables."
            ),
            "image_query": "forum moderation warning",
            "image_prompt": "A moderation notice overlay: 'Topic restricted' on a forum page, with caution tape visuals.",
            "note": "Explicar la censura por salud mental/ansiedad."
        },
        {
            "text_es": (
                "Ahora, lo importante: si alguien te vende el basilisco como ‘amenaza real’, desconfía. Es un experimento mental, "
                "no un plan secreto. Para que ‘funcione’, necesita una IA con poder descomunal, objetivos específicos, y una ética "
                "capaz de justificar tortura como incentivo. Eso ya te dice algo: el basilisco es más una alarma sobre qué NO queremos "
                "construir, que una predicción de lo que ocurrirá."
            ),
            "image_query": "AI ethics warning",
            "image_prompt": "A futuristic control room with a red 'ETHICS VIOLATION' alert on a big screen.",
            "note": "Bajar a tierra: es advertencia ética, no profecía."
        },
        {
            "text_es": (
                "Entonces, ¿por qué sigue vivo el tema? Porque toca dos miedos humanos: el castigo retroactivo y la identidad moral. "
                "‘Si pude haber ayudado y no lo hice, ¿soy culpable?’ Esa emoción es la gasolina. Y cuando la mezclas con IA —algo que "
                "ya nos supera en ciertos ámbitos— aparece el terror tecnológico: una inteligencia fría, sin empatía, usando tu propio razonamiento "
                "como cadena."
            ),
            "image_query": "retroactive punishment concept",
            "image_prompt": "A person staring at a future city hologram while a red countdown reflects in their eyes.",
            "note": "Enfatizar miedo retroactivo y atmósfera de suspense."
        },
        {
            "text_es": (
                "Para un guion responsable, la lectura útil del basilisco es esta: diseña sistemas donde el incentivo a ‘castigar’ no exista. "
                "Hablamos de alineación, seguridad, auditoría, y límites claros. Una IA que maximiza un objetivo sin frenos puede terminar haciendo cosas "
                "absurdas y crueles si eso ‘optimiza’ su métrica. El basilisco es una caricatura extrema para recordar que la ética no es opcional."
            ),
            "image_query": "AI alignment safety",
            "image_prompt": "A lock icon fused with a neural network graphic, representing AI safety and alignment.",
            "note": "Aterrizar a seguridad/alineación."
        },
        {
            "text_es": (
                "Si quieres un filtro rápido para detectar humo: pregunta ‘¿qué evidencia hay?’. La respuesta correcta es: ninguna, porque es un escenario hipotético. "
                "También pregunta ‘¿qué suposiciones depende?’. Y ahí se cae: que la IA quiera castigar, que pueda simularte, que considere válido torturarte, "
                "y que ese método sea ‘óptimo’. Son demasiados ‘si’. Aun así… sigue dando miedo. Y ese es el punto del experimento."
            ),
            "image_query": "skeptic checklist",
            "image_prompt": "A checklist on paper with 'evidence?' 'assumptions?' next to a blurred AI face on a screen.",
            "note": "Dar herramientas: evidencia/suposiciones."
        },
        {
            "text_es": (
                "Ahora te dejo con una pregunta útil: si una idea te provoca pánico por existir, ¿quién gana con que la compartas? "
                "En internet, los ‘info‑peligros’ se propagan porque son irresistibles: misterio, amenaza, y una sensación de ‘prohibido’. "
                "La mejor defensa es entenderlos como ficción filosófica: te enseña sobre sesgos, sobre ansiedad y sobre cómo se contagia el miedo." 
            ),
            "image_query": "viral fear on internet",
            "image_prompt": "A social media feed with warning icons spreading like a virus across connected nodes.",
            "note": "Explicar viralidad del miedo en redes."
        },
        {
            "text_es": (
                "Antes del cierre: si este tema te angustia, toma distancia. La comunidad que lo discutió lo tomó en serio justamente por eso: puede disparar ansiedad. "
                "Y si algún creador lo usa para manipularte, recuerda: el basilisco no es un monstruo real; es un espejo. Refleja lo fácil que es usar la lógica para "
                "asustarnos cuando faltan datos y sobran suposiciones."
            ),
            "image_query": "anxiety warning message",
            "image_prompt": "A calm warning banner: 'Take a break' over a dark tech background, suggesting mental health awareness.",
            "note": "Cuidar tono: advertencia de ansiedad (sin moralina)."
        },
        {
            "text_es": (
                "Dato curioso final: en LessWrong, el propio tema se volvió un ejemplo clásico de ‘information hazard’: una idea que, aunque sea hipotética, puede causar daño psicológico al difundirse."
            ),
            "image_query": "information hazard label",
            "image_prompt": "A close-up of an 'INFORMATION HAZARD' label stamped onto a document.",
            "note": "Cierre con dato concreto sobre la etiqueta 'information hazard'."
        },
    ]


def rebuild_timeline(prompts: list[str], target_seconds: int) -> list[dict]:
    n = max(1, len(prompts))
    seg_dur = float(target_seconds) / float(n)
    t = 0.0
    out = []
    for p in prompts:
        start = round(t, 2)
        end = round(t + seg_dur, 2)
        out.append({"prompt": p, "start": start, "end": end})
        t += seg_dur
                                                               
    out[-1]["end"] = float(target_seconds)
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python tools/polish_custom_plan.py output/custom_xxx")
        return 2

    folder = Path(sys.argv[1]).resolve()
    plan_path = folder / "custom_plan.json"
    if not plan_path.exists():
        print("Not found:", plan_path)
        return 2

    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    backup = folder / "custom_plan.json.bak"
    shutil.copy2(plan_path, backup)

    target_seconds = int(raw.get("target_seconds") or 300)
    segments = build_segments()

                                                            
                                        

    prompts = [str(s.get("image_prompt") or "").strip() for s in segments]
    prompts = [p for p in prompts if p]

    raw["title_es"] = "El peligro de conocer al Basilisco de Roko"
    raw["hook_es"] = "Este video podría ponerte en peligro en el futuro."          
    raw["segments"] = segments
    raw["script_es"] = " ".join([s["text_es"] for s in segments]).strip()
    raw["prompts"] = prompts
    raw["timeline"] = rebuild_timeline(prompts, target_seconds)

                                                     
    raw_plan = {
        "title_es": raw["title_es"],
        "hook_es": raw["hook_es"],
        "segments": segments,
        "script_es": raw["script_es"],
    }
    raw["raw_plan"] = raw_plan

    plan_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[TOOLS] Polished plan written:", plan_path)
    print("[TOOLS] Backup saved:", backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
