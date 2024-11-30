set -e

pre-commit install
git submodule update --init --recursive
