OUT_DIR=bin

$(OUT_DIR)/perceptron: $(OUT_DIR) src/main.c
	gcc -o $@ src/main.c -Wall -Wextra -ggdb -lraylib -lm

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)