#!/bin/bash
set -e
./minizero/scripts/start-container.sh -v $(pwd):/strength-detection --image kds285/strength-detection $@
