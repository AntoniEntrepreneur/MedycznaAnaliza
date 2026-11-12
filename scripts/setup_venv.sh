#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
    else
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
fi

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${VENV_DIR}/bin/python" -r requirements.txt
else
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    "${VENV_DIR}/bin/python" -m pip install -r requirements.txt
fi
