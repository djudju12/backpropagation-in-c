OUT_DIR=bin

$(OUT_DIR)/rna: $(OUT_DIR) $(OUT_DIR)/training.o src/main.c
	gcc -o $@ src/main.c -Wall -Wextra -ggdb -lraylib -lm $(OUT_DIR)/training.o

$(OUT_DIR)/training.o: src/training.c
	gcc -O2 -c -o $@ src/training.c -Wall -Wextra -ggdb

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)