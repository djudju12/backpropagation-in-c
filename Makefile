OUT_DIR=bin

CFLAGS=-Wall -Wextra -ggdb
LDFLAGS=-lraylib -lm

all: $(OUT_DIR)/guess $(OUT_DIR)/training $(OUT_DIR)/spectogram2ubyte

$(OUT_DIR)/guess: $(OUT_DIR) $(OUT_DIR)/training.o src/guess.c
	gcc src/guess.c $(CFLAGS) $(LDFLAGS) $(OUT_DIR)/training.o -o $@

$(OUT_DIR)/training: $(OUT_DIR) $(OUT_DIR)/training.o src/ui_training.c
	gcc src/ui_training.c $(CFLAGS) $(LDFLAGS) $(OUT_DIR)/training.o -o $@

$(OUT_DIR)/spectogram2ubyte: $(OUT_DIR) src/spectogram2ubyte.c
	gcc src/spectogram2ubyte.c $(CFLAGS) -lm -o $@

$(OUT_DIR)/training.o: src/training.c
	gcc -O2 -c src/training.c $(CFLAGS) -ftree-vectorize -march=native -o $@

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
