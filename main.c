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
#define RADIUS 35
#define IMAGE_SIZE 28
#define WINDOW_SIZE IMAGE_SIZE*IMAGE_SIZE

static bool pixels[WINDOW_SIZE][WINDOW_SIZE];

#define LEARNING_RATE 0.5
#define GAMMA 0.02
#define TOTAL_ITERATIONS 50

typedef struct {
    double *ws;
    uint32_t wcount;
    uint8_t label;
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
    Data *data;
    pthread_mutex_t *mut;
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

    static unsigned char metadata_buffer[16];
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

double sigmoid(double sum) {
    return 1 / (1 + exp(-sum));
}

double sum_weights(uint8_t *image, Perceptron p) {
    double sum = 0;
    for (uint32_t j = 0; j < p.wcount; j++) {
        sum += p.ws[j] * (image[j] / 255.0);
    }

    sum += p.ws[p.wcount];

    return sum;
}

double gradient_output(double sum) {
    double s = sigmoid(sum);
    return s * (1-s);
}

double generate_output(double sum) {
    return sigmoid(sum);
}

int find_label(uint8_t *image, Perceptron *ps, uint32_t ps_cnt) {
    double max = 0;
    uint8_t label = -1;
    for (size_t i = 0; i < ps_cnt; i++) {
        double y = generate_output(sum_weights(image, ps[i]));
        if (y > max) {
            label = ps[i].label;
            max = y;
        }
    }

    return label;
}

#define PERCEPTRON_CNT 10
Perceptron *unqueue(Perceptron *queue, pthread_mutex_t *mut) {
    pthread_mutex_lock(mut);
    static size_t head = 0;
    if (head >= PERCEPTRON_CNT) {
        pthread_mutex_unlock(mut);
        return NULL;
    }

    size_t my_perceptron = head;
    head += 1;

    pthread_mutex_unlock(mut);
    return &queue[my_perceptron];
}

void *train_perceptron(void *args) {
    Thread_Data *td = (Thread_Data *)args;
    for (Perceptron *p = unqueue(td->ps, td->mut); p != NULL; p = unqueue(td->ps, td->mut)) {
        for (size_t it = 0; it < TOTAL_ITERATIONS && p->stop_cond > GAMMA; it++) {
            float sum_stop_cond = 0;
            for (size_t i = 0; i < td->data->meta.size; i++) {
                size_t total_pixels = td->data->meta.rows*td->data->meta.cols;
                size_t image_index = (i * total_pixels);
                double sum = sum_weights(td->data->images + image_index, *p);
                double y = generate_output(sum);
                double gout = gradient_output(sum);
                double dy = p->label == td->data->labels[i] ? 1.0 : 0.0;
                double local_error = dy - y;
                sum_stop_cond += fabs(local_error);
                for (uint32_t j = 0; j < total_pixels && j < p->wcount; j++) {
                    double i = td->data->images[image_index + j] / 255.0;
                    p->ws[j] += LEARNING_RATE * local_error * i * gout;
                }

                p->ws[p->wcount] += LEARNING_RATE * local_error * gout;
            }

            p->stop_cond = sum_stop_cond / td->data->meta.size;
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
        printf("GAMMA: %.4f | ", GAMMA);
        for (size_t i = 0; i < 10; i++) {
            printf("[%ld]: %.4f ", i, ps[i].stop_cond);
        }

        printf("\n\r");
    }

    printf("\n");
    return NULL;
}

static const char magic[] = "PiPi";
bool dump_perceptrons(char *out, Perceptron *ps) {
    FILE *f = fopen(out, "wb");
    if (f == NULL) {
        printf("ERROR: cannot open file %s\n", out);
        goto ERROR;
    }

    if (fwrite(magic, 1, ARR_SIZE(magic) - 1, f) == 0) {
        printf("ERROR: could not write to file %s\n", out);
        goto ERROR;
    }

    for (size_t i = 0; i < 10; i++) {
        if (fwrite(&ps[i].label, sizeof(ps[i].label), 1, f) == 0) {
            printf("ERROR: could not write to file %s\n", out);
            goto ERROR;
        }

        if (fwrite(ps[i].ws, sizeof(*ps[i].ws), ps[i].wcount + 1, f) == 0) {
            printf("ERROR: could not write to file %s\n", out);
            goto ERROR;
        }
    }

    return true;
ERROR:
    if(f) fclose(f);
    return false;
}

bool load_perceptrons(char *in, Perceptron *ps) {
    FILE *f = fopen(in, "rb");
    if (f == NULL) {
        printf("ERROR: cannot open file %s\n", in);
        goto ERROR;
    }

    static unsigned char magic_buffer[4];
    if (fread(&magic_buffer, sizeof(magic_buffer), 1, f) == 0) {
        printf("cannot read file %s\n", in);
        goto ERROR;
    }

    assert(memcmp(magic, magic_buffer, 4) == 0 && "not a perceptron file");
    for (size_t i = 0; i < 10; i++) {
        ps[i].wcount = 28*28;
        ps[i].ws = malloc((ps[i].wcount + 1) * sizeof(*ps[i].ws));
        ps[i].ws[ps[i].wcount] = 0;
        ps[i].stop_cond = 1;

        if (fread(&ps[i].label, sizeof(ps[i].label), 1, f) == 0) {
            printf("ERROR: could not read label in file %s\n", in);
            goto ERROR;
        }

        if (fread(ps[i].ws, sizeof(*ps[i].ws), ps[i].wcount + 1, f) == 0) {
            printf("ERROR: could not read label in file %s\n", in);
            goto ERROR;
        }
    }

    fclose(f);
    return true;

ERROR:
    if(f) fclose(f);
    for (size_t i = 0; i < 10; i++) {
        if (ps[i].ws) free(ps[i].ws);
    }
    return false;
}

int main(void) {
    static char *images_file_path = "./data/train-images.idx3-ubyte";
    static char *labels_file_path = "./data/train-labels.idx1-ubyte";
    Data data = {0};
    read_data(images_file_path, labels_file_path, &data);

    Perceptron p[10] = { 0 };
    // uint32_t wcount = data.meta.rows * data.meta.cols;
    // for (size_t i = 0; i < 10; i++) {
    //     p[i].wcount = wcount;
    //     p[i].ws = calloc(wcount + 1, sizeof(*p[i].ws));
    //     p[i].label = i;
    //     p[i].stop_cond = 1;
    //     p[i].ws[wcount] = 0;
    // }

    if (!load_perceptrons("out.bin", p)) {
        return 1;
    }

#if 0
    long number_of_processors = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t threads[number_of_processors];

    pthread_mutex_t perceptron_mut = PTHREAD_MUTEX_INITIALIZER;
    Thread_Data td = {0};
    td.data = &data;
    td.ps = p;
    td.mut = &perceptron_mut;

    for (int i = 0; i < number_of_processors; i++) {
        pthread_create(&threads[i], NULL, train_perceptron, (void*)&td);
    }

    // pthread_t print_thread;
    // pthread_create(&print_thread, NULL, print_perceptons, (void*)p);

    for (int i = 0; i < number_of_processors; i++) {
        pthread_join(threads[i], NULL);
    }

    // training_finished = true;
    // pthread_join(print_thread, NULL);

    if (!dump_perceptrons("out.bin", p)) {
        return 1;
    }

    // return 0;
#endif
#if 0
    InitWindow(WINDOW_SIZE, WINDOW_SIZE, "Let Me Guess!");

    while (!WindowShouldClose()) {
        BeginDrawing();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            mark_mouse_pos();
        } else if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            unmark_mouse_pos();
        } else if (IsKeyPressed(KEY_SPACE))  {
            uint8_t *image = make_image();
            uint8_t label = find_label(image, p, 10);
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
        uint8_t label = find_label(image, p, 10);
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