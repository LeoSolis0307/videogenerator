import sys

if sys.platform != "win32":
    raise SystemExit("Este script solo funciona en Windows.")

import winreg


JORGE_TOKEN = "TTS_MS_ES-MX_JORGE_11.0"
ONECORE_BASE = r"SOFTWARE\\Microsoft\\Speech_OneCore\\Voices\\Tokens"
SAPI5_BASE = r"SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens"


def _copy_tree(src_root, src_path, dst_root, dst_path):
                   
    winreg.CreateKey(dst_root, dst_path)

    with winreg.OpenKey(src_root, src_path, 0, winreg.KEY_READ) as src_key:
        with winreg.OpenKey(dst_root, dst_path, 0, winreg.KEY_SET_VALUE | winreg.KEY_CREATE_SUB_KEY) as dst_key:
                            
            index = 0
            while True:
                try:
                    name, value, vtype = winreg.EnumValue(src_key, index)
                except OSError:
                    break
                winreg.SetValueEx(dst_key, name, 0, vtype, value)
                index += 1

                              
            sub_index = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(src_key, sub_index)
                except OSError:
                    break
                _copy_tree(src_root, src_path + r"\\" + sub_name, dst_root, dst_path + r"\\" + sub_name)
                sub_index += 1


def main():
    onecore_key_path = ONECORE_BASE + r"\\" + JORGE_TOKEN
    sapi5_key_path = SAPI5_BASE + r"\\" + JORGE_TOKEN

    print("[HABILITAR] Buscando Jorge en OneCore...")
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, onecore_key_path, 0, winreg.KEY_READ):
            pass
    except FileNotFoundError:
        print("❌ No encuentro la clave OneCore de Jorge:")
        print(f"   HKLM\\{onecore_key_path}")
        print("Asegúrate de tener instalada la voz 'Microsoft Jorge' (es-MX) en Windows.")
        return 1

    print("[HABILITAR] Verificando si ya existe en SAPI5...")
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, sapi5_key_path, 0, winreg.KEY_READ):
            print("✅ Ya existe en SAPI5. Debería aparecer en pyttsx3.")
            return 0
    except FileNotFoundError:
        pass

    print("[HABILITAR] Copiando token de OneCore -> SAPI5...")
    try:
        _copy_tree(winreg.HKEY_LOCAL_MACHINE, onecore_key_path, winreg.HKEY_LOCAL_MACHINE, sapi5_key_path)
    except PermissionError:
        print("❌ Permiso denegado escribiendo en HKLM.")
        print("Ejecuta PowerShell como Administrador y corre:")
        print("  C:/Users/Leonardo/Downloads/video/.venv/Scripts/python.exe C:/Users/Leonardo/Downloads/video/habilitar_jorge_sapi5.py")
        return 2

    print("✅ Copia realizada.")
    print("Ahora ejecuta:")
    print("  C:/Users/Leonardo/Downloads/video/.venv/Scripts/python.exe C:/Users/Leonardo/Downloads/video/ver_voces.py")
    print("y verifica que aparezca Jorge en la sección SAPI5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
