param(
    [string]$CondaExe = "conda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Prefix = Join-Path $ProjectRoot ".venv-train"

if (-not (Get-Command $CondaExe -ErrorAction SilentlyContinue)) {
    throw "conda.exe not found: $CondaExe"
}

if (-not (Test-Path -LiteralPath (Join-Path $Prefix "python.exe"))) {
    & $CondaExe create --prefix $Prefix python=3.10 pip -y
}

& (Join-Path $Prefix "python.exe") -m pip install `
    torch==2.8.0 torchvision==0.23.0 `
    --index-url https://download.pytorch.org/whl/cu126
& (Join-Path $Prefix "python.exe") -m pip install --no-deps -e $ProjectRoot

