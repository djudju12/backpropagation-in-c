#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    struct {
        uint32_t size;
        uint32_t rows;
        uint32_t cols;
    } meta;

    double *_images;
    uint8_t *labels;
} Data;

#define LAYER_COUNT 3
#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))

typedef struct {
    double **weights[LAYER_COUNT];
    double *values[LAYER_COUNT];
    size_t weights_count[LAYER_COUNT];
    size_t neuron_cnt[LAYER_COUNT];
    double *errors[LAYER_COUNT];
} Model;

uint32_t big2lit(unsigned char *buffer) {
    return (uint32_t) buffer[3] | (uint32_t) buffer[2] << 8 | (uint32_t) buffer[1] << 16 | (uint32_t) buffer[0] << 24;
}

bool read_data(const char *images_file_path, const char *labels_file_path, Data *data) {
    FILE *labels_file = fopen(labels_file_path, "rb");
    FILE *images_file = fopen(images_file_path, "rb");
    if (images_file == NULL) {
        fprintf(stderr, "ERROR: cannot open file '%s'\n", images_file_path);
        goto ERROR;
    }

    static unsigned char metadata_buffer[16];
    if (fread(&metadata_buffer, sizeof(metadata_buffer), 1, images_file) == 0) {
        fprintf(stderr, "ERROR: cannot read file '%s'\n", images_file_path);
        goto ERROR;
    }

    if (big2lit(metadata_buffer) != 2051) {
        fprintf(stderr, "ERROR: data file '%s' dont have correct magic number\n", images_file_path);
        goto ERROR;
    }

    data->meta.size = big2lit(metadata_buffer + 4);
    data->meta.rows = big2lit(metadata_buffer + 8);
    data->meta.cols = big2lit(metadata_buffer + 12);

    size_t total_pixels = data->meta.size * data->meta.rows * data->meta.cols;
    uint8_t *images = malloc(total_pixels * sizeof(*images));
    assert(images != NULL);
    if (fread(images, data->meta.rows*data->meta.cols, data->meta.size, images_file) != data->meta.size) {
        fprintf(stderr, "ERROR: cannot read all images from file %s\n", images_file_path);
        goto ERROR;
    }

    data->_images = malloc(total_pixels * sizeof(*data->_images));
    assert(data->_images != NULL);
    for (size_t i = 0; i < total_pixels; i++) {
        data->_images[i] = images[i] / 255.0;
    }

    free(images);

    if (labels_file == NULL) {
        fprintf(stderr, "ERROR: cannot open file %s\n", labels_file_path);
        goto ERROR;
    }

    if (fread(&metadata_buffer, sizeof(uint32_t)*2, 1, labels_file) == 0) {
        fprintf(stderr, "ERROR: cannot read file %s\n", labels_file_path);
        goto ERROR;
    }

    if (big2lit(metadata_buffer) != 2049) {
        fprintf(stderr, "ERROR: labels file '%s' dont have correct magic number\n", images_file_path);
        goto ERROR;
    }

    assert(big2lit(metadata_buffer + 4) == data->meta.size);

    data->labels = malloc(data->meta.size * sizeof(*data->labels));
    assert(data->labels != NULL);
    if (fread(data->labels, sizeof(*data->labels), data->meta.size, labels_file) != data->meta.size) {
        fprintf(stderr, "EROR: cannot read all image labels from file %s\n", images_file_path);
        goto ERROR;
    }

    fclose(labels_file);
    fclose(images_file);

    return true;

ERROR:
    if (data->_images) free(data->_images);
    if (data->labels) free(data->labels);
    if (labels_file) fclose(labels_file);
    if (images_file) fclose(images_file);

    return false;
}

#define LR 0.3

double sigmoid(double sum) {
    return .5 * (sum / (1 + fabs(sum)) + 1);
}

double sum_weights(double *weights, double *values, size_t cnt) {
    double sum = 0;
    for (uint32_t i = 0; i < cnt; i++) {
        sum += weights[i] * values[i];
    }

    sum += weights[cnt]; // bias

    return sum;
}

void train_model(Model *model, Data *training_data) {
    const size_t total_pixels = training_data->meta.rows*training_data->meta.cols;

    for (size_t _n = 0; _n < 10; _n++) {
        for (size_t i = 0; i < training_data->meta.size; i++) {
            size_t image_index = (i * total_pixels);
            int8_t label = training_data->labels[i];

            double *values = training_data->_images + image_index;
            for (size_t layer = 0; layer < LAYER_COUNT; layer++) {

                for (size_t neuron = 0; neuron < model->neuron_cnt[layer]; neuron++) {
                    double sum = sum_weights(model->weights[layer][neuron], values, model->weights_count[layer]);
                    model->values[layer][neuron] = sigmoid(sum);
                }

                values = model->values[layer];
            }

            const size_t out_layer = LAYER_COUNT - 1;
            for (int out_neuron = 0; out_neuron < (int) model->neuron_cnt[out_layer]; out_neuron++) {
                double desired = out_neuron == label ? 1. : 0.;
                double y = model->values[out_layer][out_neuron];
                double error = (desired - y) * y * (1.0 - y);
                for (size_t w_index = 0; w_index < model->weights_count[out_layer]; w_index++) {
                    double delta = LR * error * model->values[out_layer - 1][w_index];
                    model->weights[out_layer][out_neuron][w_index] += delta;
                }

                model->weights[out_layer][out_neuron][model->weights_count[out_layer]] += LR*error;
                model->errors[out_layer][out_neuron] = error;
            }

            for (int layer = out_layer - 1; layer >= 0; layer--) {
                for (size_t neuron = 0; neuron < model->neuron_cnt[layer]; neuron++) {
                    double error_sum = 0.0;

                    for (size_t next_neuron = 0; next_neuron < model->neuron_cnt[layer + 1]; next_neuron++) {
                        double error = model->errors[layer + 1][next_neuron];
                        double w = model->weights[layer + 1][next_neuron][neuron];
                        error_sum += w*error;
                    }

                    double y = model->values[layer][neuron];
                    double error = error_sum * y * (1.0 - y);

                    double *values_ = layer == 0 ? (training_data->_images + image_index) : model->values[layer - 1];
                    for (size_t w_index = 0; w_index < model->weights_count[layer]; w_index++) {
                        model->weights[layer][neuron][w_index] += LR * error * values_[w_index];
                    }

                    model->weights[layer][neuron][model->weights_count[layer]] += LR * error;
                    model->errors[layer][neuron] = error;
                }
            }
        }
    }
}

int find_label(Model *model, double *image) {
    double *values = image;
    for (size_t layer = 0; layer < LAYER_COUNT; layer++) {
        for (size_t neuron = 0; neuron < model->neuron_cnt[layer]; neuron++) {
            double sum = sum_weights(model->weights[layer][neuron], values, model->weights_count[layer]);
            model->values[layer][neuron] = sigmoid(sum);
        }

        values = model->values[layer];
    }

    int label = 0;
    for (size_t out_neuron = 1; out_neuron < model->neuron_cnt[LAYER_COUNT - 1]; out_neuron++) {
        if (model->values[LAYER_COUNT - 1][out_neuron] > model->values[LAYER_COUNT - 1][label]) {
            label = out_neuron;
        }
    }

    return label;
}

void print_results(int total, int correct) {
    // +----------------------------------+
    // | Model Statistics                 |
    // +------------------------+---------+
    // | Correct Predictions    |  8905   |
    // | Incorrect Predictions  |  1095   |
    // | Accuracy               |  89.05% |
    // | Error                  |  10.95% |
    // +------------------------+---------+

    int incorrect_guesses = total - correct;
    float acc = (correct/(double)total)*100;
    float err = (incorrect_guesses/(double)total)*100;

    printf("+----------------------------------+\n");
    printf("| Model Statistics                 |\n");
    printf("+------------------------+---------+\n");
    printf("| Correct Predicitions   |  %04d   |\n", correct);
    printf("| Incorrect Predicitions |  %04d   |\n", incorrect_guesses);
    printf("| Accuracy               |  %05.2f%% |\n", acc);
    printf("| Error                  |  %05.2f%% |\n", err);
    printf("+------------------------+---------+\n");
}

void print_image(double *image, size_t size) {
    static char alphabet[] = { '.', ':', '-', '=', '+', '*', '#', '%', '@' };

    for (size_t y = 0; y < size; y++) {
        for (size_t x = 0; x < size; x++) {
            uint8_t v = image[size*y + x] * 255.0;
            printf("%c", alphabet[(v * ARR_SIZE(alphabet)) / 256]);
        }

        printf("\n");
    }
}

bool test_model(Model *model) {
    Data testing_data = {0};
    if (!read_data("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &testing_data)) {
    // if (!read_data("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", &testing_data)) {
        return false;
    }

    int correct_guesses = 0;
    for (uint32_t i = 0; i < testing_data.meta.size; i++) {
        double *image = testing_data._images + (i * testing_data.meta.cols * testing_data.meta.rows);
        uint8_t label = find_label(model, image);
        if (label == testing_data.labels[i]) correct_guesses++;
        // printf("Guessed: %d Actual: %d\n", label, testing_data.labels[i]);
        // print_image(image, testing_data.meta.cols);
    }

    print_results(testing_data.meta.size, correct_guesses);
    return true;
}


double rand_w(float min, float max) {
    return min + (float)rand()/(float)RAND_MAX*(max-min);
}

// Make sure to the the `neuron_cnt` in the model before initialization
void init_model(Model *model, Data *data) {
    for (size_t layer = 0; layer < LAYER_COUNT; layer++) {
        int total_neurons = model->neuron_cnt[layer];
        // allocation of n neurons
        model->values[layer] = malloc(sizeof(*model->values[layer]) * total_neurons);
        assert(model->values[layer] != NULL);

        model->weights[layer] = malloc(sizeof(*model->weights) * total_neurons); // total_neurons + bias
        assert(model->weights[layer] != NULL);

        model->errors[layer] = calloc(total_neurons, sizeof(double));
        assert(model->errors[layer] != NULL);

        // set the weight count for each neuron in the layer'th layer
        if (layer == 0) {
            model->weights_count[layer] = data->meta.cols*data->meta.rows;
        } else {
            model->weights_count[layer] = model->neuron_cnt[layer - 1];
        }

        // allocation of each neuron weights
        for (int neuron_index = 0; neuron_index < total_neurons; neuron_index++) {
            model->weights[layer][neuron_index] = malloc(sizeof(double) * (model->weights_count[layer] + 1)); // + 1 for bias
            assert(model->weights[layer][neuron_index] != NULL);

            for (size_t w = 0; w < model->weights_count[layer]; w++) {
                model->weights[layer][neuron_index][w] = rand_w(-0.5, 0.5);
            }

            model->weights[layer][neuron_index][model->weights_count[layer]] = rand_w(-0.5, 0.5);
        }
    }
}

int main(void) {
    srand(time(NULL));
    static const char *images_file_path = "./data/train-images.idx3-ubyte";
    static const char *labels_file_path = "./data/train-labels.idx1-ubyte";
    Data data = {0};
    if (!read_data(images_file_path, labels_file_path, &data)) {
        return 1;
    }

    // Model model = { .neuron_cnt = {64, 32, 10} };
    Model model = { .neuron_cnt = {64, 32, 10} };
    init_model(&model, &data);
    train_model(&model, &data);
    test_model(&model);
    return 0;
}