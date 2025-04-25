#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>

#include <raylib.h>

#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))
#define LEARNING_RATE 0.1
#define RADIUS 20
#define IMAGE_SIZE 28
#define WINDOW_SIZE IMAGE_SIZE*IMAGE_SIZE

static bool pixels[WINDOW_SIZE][WINDOW_SIZE];

#define GAMMA 0.05
typedef struct {
    double *ws;
    uint32_t wcount;
    int label;
    float stop_cond;
} Perceptron;

typedef struct {
    struct {
        uint32_t size;
        uint32_t rows;
        uint32_t cols;
    } meta;

    uint8_t *images;
    uint8_t *labels;
} Data;

typedef struct {
    Perceptron *ps;
    size_t offset;
    size_t cnt;
    Data *data;
} Thread_Data;

void print_image2(uint8_t *image, uint32_t rows, uint32_t cols) {
    static char alphabet[] = { '.', ':', '-', '=', '+', '*', '#', '%', '@' };

    for (size_t y = 0; y < rows; y++) {
        for (size_t x = 0; x < cols; x++) {
            uint8_t v = image[cols*y + x];
            printf("%c", alphabet[(v * ARR_SIZE(alphabet)) / 256]);
        }

        printf("\n");
    }
}

uint32_t big2lit(unsigned char *buffer) {
    return (uint32_t) buffer[3] | (uint32_t) buffer[2] << 8 | (uint32_t) buffer[1] << 16 | (uint32_t) buffer[0] << 24;
}

bool read_data(const char *images_file_path, const char *labels_file_path, Data *data) {
    FILE *labels_file = fopen(labels_file_path, "rb");
    FILE *images_file = fopen(images_file_path, "rb");
    if (images_file == NULL) {
        printf("cannot open file %s\n", images_file_path);
        goto CLEAN_UP;
    }

    unsigned char metadata_buffer[16];
    if (fread(&metadata_buffer, sizeof(metadata_buffer), 1, images_file) == 0) {
        printf("cannot read file %s\n", images_file_path);
        goto CLEAN_UP;
    }

    assert(big2lit(metadata_buffer) == 2051);
    data->meta.size  = big2lit(metadata_buffer + 4);
    data->meta.rows  = big2lit(metadata_buffer + 8);
    data->meta.cols  = big2lit(metadata_buffer + 12);

    data->images = malloc(data->meta.size * data->meta.rows * data->meta.cols * sizeof(*data->images));
    assert(data->images != NULL);
    if (fread(data->images, data->meta.rows*data->meta.cols, data->meta.size, images_file) != data->meta.size) {
        printf("cannot read all images from file %s\n", images_file_path);
        goto CLEAN_UP;
    }

    if (labels_file == NULL) {
        printf("cannot open file %s\n", labels_file_path);
        goto CLEAN_UP;
    }

    if (fread(&metadata_buffer, sizeof(uint32_t)*2, 1, labels_file) == 0) {
        printf("cannot read file %s\n", labels_file_path);
        goto CLEAN_UP;
    }

    assert(big2lit(metadata_buffer) == 2049);
    assert(big2lit(metadata_buffer + 4) == data->meta.size);

    data->labels = malloc(data->meta.size * sizeof(*data->labels));
    assert(data->labels != NULL);
    if (fread(data->labels, sizeof(*data->labels), data->meta.size, labels_file) != data->meta.size) {
        printf("cannot read all images from file %s\n", images_file_path);
        goto CLEAN_UP;
    }

    fclose(labels_file);
    fclose(images_file);

    return true;

CLEAN_UP:
    if (data->images) free(data->images);
    if (data->labels) free(data->labels);
    if (labels_file) fclose(labels_file);
    if (images_file) fclose(images_file);

    return false;
}

bool f(double sum) {
    return sum > 0;
}

bool generate_y(Data data, Perceptron p, int i) {
    double sum = 0;
    for (uint32_t j = 0; j < data.meta.rows*data.meta.cols && j < p.wcount; j++) {
        sum += p.ws[j] * data.images[(i * data.meta.cols * data.meta.rows) + j];
    }

    return f(sum);
}

bool generate_y_from_image(uint8_t *image, uint32_t rows, uint32_t cols, Perceptron p) {
    assert(rows * cols == p.wcount);
    double sum = 0;
    for (uint32_t j = 0; j < p.wcount; j++) {
        sum += p.ws[j] * image[j];
    }

    return f(sum);
}

uint8_t find_label(uint8_t *image, uint32_t rows, uint32_t cols, Perceptron *ps, uint32_t ps_cnt) {
    for (size_t i = 0; i < ps_cnt; i++) {
        bool y = generate_y_from_image(image, rows, cols, ps[i]);
        if (y) {
            return ps[i].label;
        }
    }

    return 255;
}

void *train_perceptron(void *args) {
    Thread_Data *td = (Thread_Data *)args;
    for (size_t p = 0; p < td->cnt; p++) {
        if (td->ps[p + td->offset].stop_cond <= GAMMA) continue;
        for (size_t it = 0; it < 10; it++) {
            float sum_stop_cond = 0;
            for (size_t i = 0; i < td->data->meta.size; i++) {
                bool y = generate_y(*td->data, td->ps[p + td->offset], i);
                bool dy = td->ps[p + td->offset].label == td->data->labels[i];
                sum_stop_cond += abs(dy - y);
                if (y != dy) {
                    for (uint32_t j = 0; j < td->data->meta.rows*td->data->meta.cols && j < td->ps[p + td->offset].wcount; j++) {
                        td->ps[p + td->offset].ws[j] += LEARNING_RATE*((int) dy - (int) y)*td->data->images[(i * td->data->meta.cols * td->data->meta.rows) + j];
                    }
                }
            }

            td->ps[p + td->offset].stop_cond = sum_stop_cond / td->data->meta.size;
        }
    }

    return NULL;
}

void mark_mouse_pos(void) {
    Vector2 pos = GetMousePosition();
    if (pos.y >= 0 && pos.x >= 0 && pos.y < WINDOW_SIZE && pos.x < WINDOW_SIZE) pixels[(int)pos.y][(int)pos.x] = true;
}

void unmark_mouse_pos(void) {
    Vector2 pos = GetMousePosition();
    if (pos.y >= 0 && pos.x >= 0 && pos.y < WINDOW_SIZE && pos.x < WINDOW_SIZE) pixels[(int)pos.y][(int)pos.x] = false;
}

void paint_marks(void) {
    for (size_t y = 0; y < WINDOW_SIZE; y++) {
        for (size_t x = 0; x < WINDOW_SIZE; x++) {
            if (pixels[y][x]) {
                DrawCircle(x, y, RADIUS, WHITE);
            }
        }
    }
}

uint8_t *make_image() {
    static uint8_t buffer[IMAGE_SIZE][IMAGE_SIZE];
    Image window_image = LoadImageFromScreen();
    for (size_t iy = 0; iy < IMAGE_SIZE; iy++) {
        for (size_t ix = 0; ix < IMAGE_SIZE; ix++) {
            uint32_t sum = 0;
            for (size_t wy = 0; wy < IMAGE_SIZE; wy++) {
                for (size_t wx = 0; wx < IMAGE_SIZE; wx++) {
                    Color c = GetImageColor(window_image, (ix*IMAGE_SIZE) + wx, (iy*IMAGE_SIZE) + wy);
                    sum += (c.r + c.g + c.b)/3;
                }
            }

            buffer[iy][ix] = (sum/IMAGE_SIZE*IMAGE_SIZE);
        }
    }

    return (uint8_t*) buffer;
}

volatile int training_finished = false;
void *print_perceptons(void *args) {
    Perceptron *ps = (Perceptron *) args;
    while (!training_finished) {
        for (size_t i = 0; i < 10; i++) {
            printf("[%ld]: %.4f ", i, ps[i].stop_cond);
        }

        printf("\n");
    }

    return NULL;
}

int main(void) {
    static char *images_file_path = "./data/train-images.idx3-ubyte";
    static char *labels_file_path = "./data/train-labels.idx1-ubyte";
    Data data = {0};
    read_data(images_file_path, labels_file_path, &data);

    // print_image(data, 0);
    // print_image2(data.images + (1 * data.meta.rows * data.meta.cols), data.meta.rows, data.meta.cols);

    Perceptron p[10] = { 0 };
    uint32_t wcount = data.meta.rows * data.meta.cols;
    for (size_t i = 0; i < 10; i++) {
        p[i].wcount = wcount;
        p[i].ws = calloc(wcount, sizeof(*p[i].ws));
        p[i].label = i;
        p[i].stop_cond = 1;
    }

    long number_of_processors = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t threads[number_of_processors];
    Thread_Data threads_data[number_of_processors];
    if (number_of_processors > 10) {
        for (int i = 0; i < number_of_processors; i++) {
            threads_data[i] = (Thread_Data) {
                .cnt = 1,
                .offset = i,
                .data = &data,
                .ps = p
            };
        }
    } else {
        int per_processor = 10 / number_of_processors;
        int remaning = 10 % number_of_processors;
        int offset = 0;
        for (int i = 0; i < number_of_processors; i++) {
            int cnt = per_processor;
            if (remaning > 0) {
                if (i == number_of_processors - 1) cnt += remaning;
                else {
                    cnt++;
                    remaning--;
                }
            }

            threads_data[i] = (Thread_Data) {
                .cnt = cnt,
                .offset = offset,
                .data = &data,
                .ps = p
            };

            offset += cnt;
        }
    }

    for (int i = 0; i < number_of_processors; i++) {
        pthread_create(&threads[i], NULL, train_perceptron, (void*)&threads_data[i]);
    }

    // pthread_t print_thread;
    // pthread_create(&print_thread, NULL, print_perceptons, (void*)p);

    printf("Training...... ");

    for (int i = 0; i < number_of_processors; i++) {
        pthread_join(threads[i], NULL);
    }

    training_finished = true;
    // pthread_join(print_thread, NULL);

#if 1
    InitWindow(WINDOW_SIZE, WINDOW_SIZE, "Let Me Guess!");

    while (!WindowShouldClose()) {
        BeginDrawing();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            mark_mouse_pos();
        } else if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            unmark_mouse_pos();
        } else if (IsKeyPressed(KEY_SPACE))  {
            uint8_t *image = make_image();
            uint8_t label = find_label(image, 28, 28, p, 10);
            printf("Guessed: %u\n", label);
            print_image2(image, 28, 28);
        } else if (IsKeyPressed(KEY_R))  {
            for (size_t y = 0; y < WINDOW_SIZE; y++) {
                for (size_t x = 0; x < WINDOW_SIZE; x++) {
                    pixels[y][x] = false;
                }
            }
        }

        ClearBackground(BLACK);
        paint_marks();

        EndDrawing();
    }

    CloseWindow();

#else
    Data testing_data = {0};
    read_data("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &testing_data);

    int correct_guesses = 0;
    for (uint32_t i = 0; i < testing_data.meta.size; i++) {
        uint8_t *image = testing_data.images + (i * testing_data.meta.cols * testing_data.meta.rows);
        uint8_t label = find_label(image, 28, 28, p, 10);
        // printf("[%d] Guessed: %d. Actual: %d\n", i, label, testing_data.labels[i]);
        // print_image2(image, 28, 28);
        if (label == testing_data.labels[i]) correct_guesses++;
    }

    int incorrect_guesses = testing_data.meta.size - correct_guesses;
    printf("%04d/%d\n", correct_guesses, testing_data.meta.size);
    printf("%04d/%d\n", incorrect_guesses, testing_data.meta.size);
    printf("%.2f%%\n", (correct_guesses/(double)testing_data.meta.size)*100);
    printf("%.2f%%\n", (incorrect_guesses/(double)testing_data.meta.size)*100);
#endif
    return 0;
}