#!/bin/bash

for dir in [0-9abcdef]*; do
    if [ -d "$dir" ]; then
        echo "Delete: $dir"
        rm -rf "$dir"
    fi
done
