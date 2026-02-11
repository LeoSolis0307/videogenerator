import json
import os
import sys
import time
from pathlib import Path

                                                          
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import custom_video


def main() -> None:
    brief = (
        'Escribe un guion para YouTube sobre "El peligro de conocer al Basilisco de Roko". '
        'Tono: Terror Tecnológico. '
        'Datos: '
        '1. Es un experimento mental sobre una futura súper IA que castiga a quien no ayudó a crearla. '
        '2. Se considera un "info-peligro": solo saber que existe la teoría te pone en su lista negra. '
        '3. Muchos foros de tecnología prohibieron hablar del tema por pánico real. '
        'Gancho: "Este video podría ponerte en peligro en el futuro"'
    )

    target_seconds = 300
    out_dir = Path('output') / f"plan_only_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[TOOLS] OLLAMA_TEXT_MODEL env =', os.environ.get('OLLAMA_TEXT_MODEL'))
    print('[TOOLS] custom_video short model =', getattr(custom_video, 'OLLAMA_TEXT_MODEL_SHORT', None))
    print('[TOOLS] custom_video long model  =', getattr(custom_video, 'OLLAMA_TEXT_MODEL_LONG', None))

    plan = custom_video.generar_plan_personalizado(brief, min_seconds=target_seconds)

    out_path = out_dir / 'custom_plan.json'
    # Convert Pydantic object to dict for serialization
    data = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[TOOLS] Saved:', out_path)


if __name__ == '__main__':
    main()
