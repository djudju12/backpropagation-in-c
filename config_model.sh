#!/bin/bash

LEARNING_RATE=0.5
MAX_ITERATIONS=1
TOLERANCE=0.02

./bin/perceptron --train --out-dir models --lr $LEARNING_RATE --max-iters $MAX_ITERATIONS --tolerance $TOLERANCE