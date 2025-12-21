from deep_translator import GoogleTranslator

def traducir_lista(textos):
    print("[TEXT] Traduciendo textos a español...")
    traducidos = []

    for i, texto in enumerate(textos):
        print(f"[TEXT] Traduciendo {i+1}/{len(textos)}")
        try:
                           
            if len(texto) > 4500:
                texto = texto[:4500]

            es = GoogleTranslator(source="auto", target="es").translate(texto)
            traducidos.append(es)
        except Exception as e:
            print(f"[TEXT] Error traduciendo: {e}")
            raise RuntimeError(f"Fallo traduciendo item {i+1}/{len(textos)}") from e

    return traducidos
