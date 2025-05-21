#!/bin/bash

LEARNING_RATE=0.02
MAX_ITERATIONS=2
TOLERANCE=0.02

./bin/training --train --out-dir models --lr $LEARNING_RATE --max-iters $MAX_ITERATIONS --tolerance $TOLERANCE

