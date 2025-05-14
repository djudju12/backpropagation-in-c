OUT_DIR=bin

$(OUT_DIR)/backpropagation: $(OUT_DIR) src/algorithms/backpropagation.c
	gcc -O2 -o $@ src/algorithms/backpropagation.c -Wall -Wextra -ggdb -lraylib -lm

$(OUT_DIR)/perceptron: $(OUT_DIR) src/main.c
	gcc -o $@ src/main.c -Wall -Wextra -ggdb -lraylib -lm

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)