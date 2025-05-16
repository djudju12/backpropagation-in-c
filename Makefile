OUT_DIR=bin

$(OUT_DIR)/rna: $(OUT_DIR) src/main.c
	gcc -O2 -o $@ src/main.c -Wall -Wextra -ggdb -lraylib -lm

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)