#!/bin/bash
set -euo pipefail

usage() {
    printf '%s\n' 'Usage: install.sh [--pixinsight-scripts DIRECTORY] [--backend EXECUTABLE]'
}

scripts=
backend=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --pixinsight-scripts|--backend)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                usage >&2
                exit 2
            fi
            if [ "$1" = --pixinsight-scripts ]; then
                scripts=$2
            else
                backend=$2
            fi
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$scripts" ]; then
    for candidate in \
        "/Applications/PixInsight/src/scripts" \
        "/Applications/PixInsight/PixInsight.app/Contents/Resources/src/scripts" \
        "/Applications/PixInsight.app/Contents/Resources/src/scripts"; do
        if [ -d "$candidate" ]; then
            scripts=$candidate
            break
        fi
    done
fi
if [ ! -d "$scripts" ]; then
    printf '%s\n' 'PixInsight scripts directory not found. Pass --pixinsight-scripts with the path to an existing directory.' >&2
    exit 1
fi
destination="$(cd -- "$scripts" && pwd -P)/DWT-Drizzle"
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
core="$(dirname -- "$source_dir")"

if [ -z "$backend" ]; then
    backend="$source_dir/bin/dwt-backend"
    if [ ! -f "$backend" ]; then
        if ! command -v cargo >/dev/null 2>&1; then
            printf '%s\n' 'Cargo not found. Install Rust from https://rustup.rs/ or pass --backend with a macOS dwt-backend.' >&2
            exit 1
        fi
        cargo build --manifest-path "$core/Cargo.toml" --target-dir "$core/target" --release --locked --bin dwt-backend
        backend="$core/target/release/dwt-backend"
    fi
fi
if [ ! -f "$backend" ]; then
    printf 'Backend executable not found at %s. Pass --backend to use a build in another location.\n' "$backend" >&2
    exit 1
fi

mkdir -p "$destination/bin"
install -m 0755 "$backend" "$destination/bin/dwt-backend"
for file in pipeline-builder.js pipeline-builder-ui.js event-script.js dwt-backend.json.example README.md; do
    install -m 0644 "$source_dir/$file" "$destination/$file"
done

printf 'Installed DWT-Drizzle WBPP scripts in %s\n' "$destination"
printf 'Select %s/pipeline-builder-ui.js in WBPP > Pipeline > Pipeline scripting > Pipeline builder.\n' "$destination"
