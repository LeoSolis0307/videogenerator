import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], *, timeout: int = 0) -> None:
    printable = " ".join(cmd)
    print(f"[SETUP-PIPER] $ {printable}")
    kwargs = {"check": True}
    if timeout and timeout > 0:
        kwargs["timeout"] = timeout
    subprocess.run(cmd, **kwargs)


def _pip_install(packages: list[str]) -> None:
    py = sys.executable or "python"
    _run([py, "-m", "pip", "install", *packages], timeout=900)


def _download_model(
    *,
    repo_id: str,
    filenames: list[str],
    local_dir: Path,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "Falta dependencia 'huggingface_hub'. Ejecuta sin --skip-install o instala con: pip install huggingface_hub"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None

    for filename in filenames:
        try:
            print(f"[SETUP-PIPER] Intentando modelo: {filename}")
            onnx_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{filename}.json",
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
            except Exception:
                pass
            return Path(onnx_path).resolve()
        except Exception as exc:
            last_err = exc
            print(f"[SETUP-PIPER] Falló candidato: {filename} ({type(exc).__name__}: {exc})")

    raise RuntimeError(f"No se pudo descargar ningún modelo candidato. Último error: {last_err}")


def _save_model_hint(path: Path) -> Path:
    hint_file = ROOT / "storage" / "piper_model_path.txt"
    hint_file.parent.mkdir(parents=True, exist_ok=True)
    hint_file.write_text(str(path), encoding="utf-8")
    return hint_file


def _smoke_test(model_path: Path, output_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "core.local_tts",
        "--text",
        "Prueba rápida de voz local con Piper.",
        "--out",
        str(output_path),
        "--engine",
        "piper",
        "--piper-model",
        str(model_path),
    ]
    _run(cmd, timeout=300)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instalador one-click de Piper para TTS local")
    parser.add_argument(
        "--model",
        default="",
        help="Ruta de voz dentro del repo rhasspy/piper-voices (ej: es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx)",
    )
    parser.add_argument(
        "--repo-id",
        default="rhasspy/piper-voices",
        help="Repositorio de Hugging Face para voces de piper",
    )
    parser.add_argument(
        "--model-dir",
        default=str(ROOT / "models" / "piper"),
        help="Directorio local para guardar modelos",
    )
    parser.add_argument("--skip-install", action="store_true", help="No instala paquetes Python")
    parser.add_argument("--skip-smoke", action="store_true", help="No ejecuta prueba de síntesis")
    parser.add_argument(
        "--out-smoke",
        default=str(ROOT / "output" / "_piper_setup_smoke.wav"),
        help="Archivo WAV de prueba",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    candidates = [
        "es/es_MX/ald/medium/es_MX-ald-medium.onnx",
        "es/es_MX/ald/low/es_MX-ald-low.onnx",
        "es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
        "es/es_ES/sharvard/low/es_ES-sharvard-low.onnx",
    ]

    if args.model.strip():
        candidates = [args.model.strip()]

    try:
        if not args.skip_install:
            _pip_install(["piper-tts", "huggingface_hub", "pathvalidate"])

        model_path = _download_model(
            repo_id=args.repo_id,
            filenames=candidates,
            local_dir=Path(args.model_dir),
        )
        hint_file = _save_model_hint(model_path)

        print(f"[SETUP-PIPER] Modelo listo: {model_path}")
        print(f"[SETUP-PIPER] Hint guardado: {hint_file}")

        if not args.skip_smoke:
            _smoke_test(model_path, Path(args.out_smoke))
            print(f"[SETUP-PIPER] Smoke OK: {args.out_smoke}")

        print("\n[SETUP-PIPER] Listo. Uso recomendado:")
        print(
            f"python -m core.local_tts --text \"Hola\" --out output/piper_demo.wav --engine piper --piper-model \"{model_path}\""
        )
        print("O simplemente (si guardaste hint):")
        print("python -m core.local_tts --text \"Hola\" --out output/piper_demo.wav --engine piper")
        return 0
    except Exception as exc:
        print(f"[SETUP-PIPER] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
