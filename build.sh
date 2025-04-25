#!/bin/bash

set -x

gcc -o perceptron main.c -Wall -Wextra -ggdb -lraylib -lm