
#!/bin/bash

FILENAME=$1

if [ -z "$FILENAME" ]; then
    exit 1
fi

if [ ! -f "$FILENAME" ]; then
    exit 1
fi

mkdir -p build

OUTPUT="${FILENAME%.cpp}"

g++ -fopenmp -O2 -Wall "$FILENAME" -o "build/$OUTPUT"
