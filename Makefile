OUT_DIR=bin

$(OUT_DIR)/rna: $(OUT_DIR) $(OUT_DIR)/training.o src/main.c
	gcc -o $@ src/main.c -Wall -Wextra -ggdb -lraylib -lm $(OUT_DIR)/training.o -pg

$(OUT_DIR)/training.o: src/training.c
	gcc -O2 -c -o $@ src/training.c -Wall -Wextra -ggdb -ftree-vectorize -march=native -pg

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)