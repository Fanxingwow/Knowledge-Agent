#!/bin/sh
set -eu

umask 027

required_writable_paths="
${IMPORT_WORK_ROOT:-/var/lib/knowledge-agent/imports}
/app/logs
${HF_HOME:-/var/cache/knowledge-agent/huggingface}
${MODELSCOPE_CACHE:-/var/cache/knowledge-agent/modelscope}
"

for path in $required_writable_paths; do
    mkdir -p "$path"
    if [ ! -w "$path" ]; then
        echo "Required runtime path is not writable: $path" >&2
        exit 1
    fi
done

exec "$@"
