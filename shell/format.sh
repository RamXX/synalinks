#!/bin/bash
set -Euo pipefail

base_dir=$(dirname $(dirname $0))

uvx black --line-length 90 synalinks/src

uvx ruff check --config "${base_dir}/pyproject.toml" --fix .

uvx ruff format --config "${base_dir}/pyproject.toml" .
