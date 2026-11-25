param(
    [string] $PixInsightScripts,
    [string] $RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path,
    [string] $BackendPath
)

$ErrorActionPreference = "Stop"

if (!$PixInsightScripts)
{
    $roots = @(
        Join-Path $env:ProgramFiles "PixInsight"
        Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
            "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*" -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -eq "PixInsight" -and $_.InstallLocation } |
            Select-Object -ExpandProperty InstallLocation
    )
    foreach ($root in $roots)
    {
        $candidate = Join-Path $root "src\scripts"
        if (Test-Path -LiteralPath $candidate -PathType Container)
        {
            $PixInsightScripts = $candidate
            break
        }
    }
}
if (!$PixInsightScripts -or !(Test-Path -LiteralPath $PixInsightScripts -PathType Container))
{
    throw "PixInsight scripts directory not found. Pass -PixInsightScripts with the path to an existing directory."
}
$destination = Join-Path (Resolve-Path -LiteralPath $PixInsightScripts).Path "DWT-Drizzle"
$core = Join-Path $RepositoryRoot "core-rs"
$source = Join-Path $core "wbpp"

if (!$BackendPath)
{
    $BackendPath = Join-Path $source "bin\dwt-backend.exe"
    if (!(Test-Path -LiteralPath $BackendPath -PathType Leaf))
    {
        if (!(Get-Command cargo -ErrorAction SilentlyContinue))
        {
            throw "Cargo not found. Install Rust from https://rustup.rs/ or pass -BackendPath with a Windows dwt-backend.exe."
        }
        $targetDir = Join-Path $core "target"
        cargo build --manifest-path (Join-Path $core "Cargo.toml") --target-dir $targetDir --release --locked --bin dwt-backend
        if ($LASTEXITCODE -ne 0)
        {
            throw "The Rust backend build failed (exit $LASTEXITCODE)."
        }
        $BackendPath = Join-Path $targetDir "release\dwt-backend.exe"
    }
}
if (!(Test-Path -LiteralPath $BackendPath -PathType Leaf))
{
    throw "Backend executable not found at '$BackendPath'. Pass -BackendPath to use a build in another location."
}

$binaryDestination = Join-Path $destination "bin"
New-Item -ItemType Directory -Path $binaryDestination -Force | Out-Null
Copy-Item -LiteralPath $BackendPath -Destination (Join-Path $binaryDestination "dwt-backend.exe") -Force
foreach ($file in @("pipeline-builder.js", "pipeline-builder-ui.js", "event-script.js", "dwt-backend.json.example", "README.md"))
{
    Copy-Item -LiteralPath (Join-Path $source $file) -Destination $destination -Force
}

Write-Output "Installed DWT-Drizzle WBPP scripts in $destination"
Write-Output "Select $(Join-Path $destination 'pipeline-builder-ui.js') in WBPP > Pipeline > Pipeline scripting > Pipeline builder."
