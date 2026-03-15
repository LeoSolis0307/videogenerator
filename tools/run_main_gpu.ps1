$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$env:VIDEO_CODEC = "h264_amf"
$env:REQUIRE_GPU = "1"
if (-not $env:VIDEO_CODEC_ARGS) {
    $env:VIDEO_CODEC_ARGS = "-quality quality -usage transcoding -rc cqp -qp_i 24 -qp_p 24"
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "No se encontró Python del entorno virtual en: $python"
}

Write-Host "[GPU] VIDEO_CODEC=$env:VIDEO_CODEC"
Write-Host "[GPU] REQUIRE_GPU=$env:REQUIRE_GPU"
Write-Host "[GPU] VIDEO_CODEC_ARGS=$env:VIDEO_CODEC_ARGS"
Write-Host "[GPU] Ejecutando main.py..."

& $python "main.py"
exit $LASTEXITCODE
