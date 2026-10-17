#!/bin/sh
# Preserve the frozen launcher's interpreter path through an image-local alias.
exec /opt/venv/bin/python3 "$@"
