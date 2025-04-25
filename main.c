#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))

typedef struct {
    struct {
        uint32_t size;
        uint32_t rows;
        uint32_t cols;
    } meta;

    uint8_t *images;
    uint8_t *labels;
} Data;

void print_image(Data data, size_t i) {
    assert(i < data.meta.size);
    static char alphabet[] = { '.', ':', '-', '=', '+', '*', '#', '%', '@' };

    printf("Number: %u\n", data.labels[i]);
    for (size_t y = 0; y < data.meta.rows; y++) {
        for (size_t x = 0; x < data.meta.cols; x++) {
            uint8_t v = data.images[(i * data.meta.cols * data.meta.rows) + data.meta.cols*y + x];
            printf("%c", alphabet[v / (255 / (ARR_SIZE(alphabet) - 1))]);
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

typedef struct {
    double *ws;
    uint32_t wcount;
    int label;
} Perceptron;

bool f(double);
bool gerenate_y(Data, Perceptron, int);

static const double lr = 0.5;

void train_perceptron(Perceptron *p, Data traning_data);
int main(void) {
    static char *images_file_path = "./data/train-images.idx3-ubyte";
    static char *labels_file_path = "./data/train-labels.idx1-ubyte";
    Data data = {0};
    read_data(images_file_path, labels_file_path, &data);

    Perceptron p[10] = { 0 };
    uint32_t wcount = data.meta.size * data.meta.rows * data.meta.cols;
    for (size_t i = 0; i < 10; i++) {
        p[i].wcount = wcount;
        p[i].ws = calloc(wcount, sizeof(*p[i].ws));
        p[i].label = i;
    }

    for (size_t k = 0; k < 10; k++) {
        train_perceptron(&p[k], data);
    }

    Data testing_data = {0};
    read_data("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &testing_data);

    int correct_guesses = 0;
    for (uint32_t i = 0; i < testing_data.meta.size; i++) {
        for (size_t j = 0; j < 10; j++) {
            bool y = gerenate_y(testing_data, p[j], i);
            bool dy = p[j].label == testing_data.labels[i];
            if (y && dy) {
                correct_guesses++;
            }
        }
    }

    printf("%d/%d\n", correct_guesses, testing_data.meta.size);
    printf("%.2f%%\n", (correct_guesses/(double)testing_data.meta.size)*100);

    return 0;
}

void train_perceptron(Perceptron *p, Data traning_data) {
    size_t total_it = 10;
    for (size_t it = 0; it < total_it; it++) {
        printf("Training perception %d: %ld/%ld\n", p->label, it, total_it);
        for (size_t i = 0; i < traning_data.meta.size; i++) {
            bool y = gerenate_y(traning_data, *p, i);
            bool dy = p->label == traning_data.labels[i];
            if (y != (p->label == traning_data.labels[i])) {
                for (uint32_t j = 0; j < traning_data.meta.rows*traning_data.meta.cols && j < p->wcount; j++) {
                    p->ws[j] += lr*((int) dy - (int) y)*traning_data.images[(i * traning_data.meta.cols * traning_data.meta.rows) + j];
                }
            }
        }
    }
}

bool gerenate_y(Data data, Perceptron p, int i) {
    double sum = 0;
    for (uint32_t j = 0; j < data.meta.rows*data.meta.cols && j < p.wcount; j++) {
        sum += p.ws[j] * data.images[(i * data.meta.cols * data.meta.rows) + j];
    }

    return f(sum);
}

bool f(double sum) {
    // y -> 0...10
    return sum > 0;
}

// bool dump_weights(char *out_path, Perceptron p) {
//     FILE *f = fopen(out_path, "wb");
//     if (f == NULL) {
//         printf("cannot open file %s\n", out_path);
//         return false;
//     }

//     static unsigned char magic[] = {'N', 'N', 'W', 'G'};
//     if (fwrite(magic, ARR_SIZE(magic), 1, f) == 0) {
//         printf("could not write to file %s\n", out_path);
//         return false;
//     }

//     if (fwrite(&p.wcount, sizeof(p.wcount), 1, f) == 0) {
//         printf("could not write to file %s\n", out_path);
//         return false;
//     }

//     if (fwrite(p.ws, sizeof(*p.ws), p.wcount, f) == 0) {
//         printf("could not write to file %s\n", out_path);
//         return false;
//     }

//     return true;
// }

// bool *load_weights(char *path, Perceptron *p) {
//     FILE *weights_file = fopen(path, "rb");
//     if (weights_file == NULL) {
//         printf("ERROR: could not open file %s\n", path);
//         goto CLEAN_UP;
//     }

//     char magic_number[4];
//     if (fread(magic_number, sizeof(magic_number), 1, weights_file) == 0) {
//         printf("ERROR: could read from file %s\n", path);
//         goto CLEAN_UP;
//     }

//     if (memcmp(magic_number, "NNWG", 4) != 0) {
//         printf("ERROR: %s its not a weights file\n", path);
//         goto CLEAN_UP;
//     }

//     fclose(weights_file);
//     return true;

// CLEAN_UP:
//     if (weights_file) fclose(weights_file);

//     return false;
// }