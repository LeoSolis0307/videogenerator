from deep_translator import GoogleTranslator

translator = GoogleTranslator(source="auto", target="es")

def traducir(texto: str) -> str:
    if len(texto) > 4500:
        texto = texto[:4500]
    try:
        return translator.translate(texto)
    except:
        return texto


def limpiar(texto: str) -> str:
    return texto.replace("\n", " ").strip()


def filtrar_comentarios(comments, historial, min_len=200):
    textos = []
    nuevos_ids = []
    for c in comments:
        if c["kind"] != "t1":
            continue
        body = c["data"].get("body", "")
        cid = c["data"].get("id")
        if cid in historial:
            continue
        if len(body) < min_len or "[deleted]" in body:
            continue
        textos.append(traducir(limpiar(body)))
        nuevos_ids.append(cid)
    return textos, nuevos_ids
