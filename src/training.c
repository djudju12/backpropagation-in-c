#include "training.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <pthread.h>

#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))
#define ERROR_VALIDATION_STEP 2500 // check error sum each ERROR_VALIDATION_STEP iterations
#define LOG_WRITE_ERROR(file) fprintf(stderr, "ERROR: could not write to file %s\n", file)
#define LOG_READ_ERROR(msg, file) fprintf(stderr, "ERROR: could not read %s from file %s\n", msg, file)
#define internal static

static RNA_Parameters default_parameters = {
    .lr = 0.5,
    .tolerance = 0.02,
    .max_iters = 50
};

RNA_Parameters get_default_parameters(void) {
    return default_parameters;
}

internal uint32_t big2lit(unsigned char *buffer) {
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

internal double sigmoid(double sum) {
    return .5 * (sum / (1 + fabs(sum)) + 1);
}

internal double dot_product(double *a, double *b, size_t cnt) {
    double sum = 0;

    for (size_t i = 0; i + 3 < cnt; i += 4) {
        sum += a[i]   * b[i]
             + a[i+1] * b[i+1]
             + a[i+2] * b[i+2]
             + a[i+3] * b[i+3];
    }

    size_t remaning = cnt % 4;
    for (size_t i = 0; i < remaning; i++) {
        sum += a[cnt - 1 - i] * b[cnt - 1 - i];
    }

    return sum;
}

internal double sum_weights(double *weights, double *values, size_t cnt) {
    double sum = dot_product(weights, values, cnt);
    sum += weights[cnt]; // bias
    return sum;
}

internal double *get_neuron_weights(RNA_Model *model, uint32_t layer, uint32_t neuron) {
    return model->weights[layer] + neuron*(model->weights_cnt[layer] + 1);
}

int find_label(RNA_Model *model, double *image) {
    double *values = image;
    for (size_t layer = 0; layer < model->layer_count; layer++) {
        for (size_t neuron = 0; neuron < model->neuron_cnt[layer]; neuron++) {
            double sum = sum_weights(
                get_neuron_weights(model, layer, neuron),
                values,
                model->weights_cnt[layer]
            );
            model->values[layer][neuron] = sigmoid(sum);
        }

        values = model->values[layer];
    }

    int label = 0;
    for (size_t out_neuron = 1; out_neuron < model->neuron_cnt[model->layer_count - 1]; out_neuron++) {
        if (model->values[model->layer_count - 1][out_neuron] > model->values[model->layer_count - 1][label]) {
            label = out_neuron;
        }
    }

    return label;
}

internal void _train_model(RNA_Model *model, Data *training_data) {
    const size_t total_pixels = training_data->meta.rows*training_data->meta.cols;
    const double lr = model->training_parameters->lr;
    model->training = true;

    double global_error = 0;
    for (int it = 0; it < model->training_parameters->max_iters; it++) {
        for (size_t i = 0; i < training_data->meta.size; i++) {
            size_t image_index = (i * total_pixels);
            int8_t label = training_data->labels[i];

            double *values = training_data->_images + image_index;
            for (size_t layer = 0; layer < model->layer_count; layer++) {
                for (size_t neuron = 0; neuron < model->neuron_cnt[layer]; neuron++) {
                    double sum = sum_weights(get_neuron_weights(model, layer, neuron), values, model->weights_cnt[layer]);
                    model->values[layer][neuron] = sigmoid(sum);
                }

                values = model->values[layer];
            }

            double local_error = 0.;
            const size_t out_layer = model->layer_count - 1;
            for (int out_neuron = 0; out_neuron < (int) model->neuron_cnt[out_layer]; out_neuron++) {
                double desired = out_neuron == label ? 1. : 0.;
                double y = model->values[out_layer][out_neuron];
                double error = (desired - y) * y * (1.0 - y);
                local_error += (desired - y) * (desired - y);
                double *weights = get_neuron_weights(model, out_layer, out_neuron);
                for (size_t w_index = 0; w_index < model->weights_cnt[out_layer]; w_index++) {
                    double delta = lr * error * model->values[out_layer - 1][w_index];
                    weights[w_index] += delta;
                }

                weights[model->weights_cnt[out_layer]] += lr * error;
                model->errors[out_layer][out_neuron] = error;
            }

            global_error += local_error*0.5;

            for (int layer = out_layer - 1; layer >= 0; layer--) {
                uint32_t neuron_cnt = model->neuron_cnt[layer];
                for (size_t neuron = 0; neuron < neuron_cnt; neuron++) {
                    double error_sum = 0.0;

                    for (size_t next_neuron = 0; next_neuron < model->neuron_cnt[layer + 1]; next_neuron++) {
                        double error = model->errors[layer + 1][next_neuron];
                        double w = get_neuron_weights(model, layer + 1, next_neuron)[neuron];
                        error_sum += w*error;
                    }

                    double y = model->values[layer][neuron];
                    double error = error_sum * y * (1.0 - y);

                    uint32_t w_count = model->weights_cnt[layer];
                    double *values_ = layer == 0 ? (training_data->_images + image_index) : model->values[layer - 1];
                    double *weights = get_neuron_weights(model, layer, neuron);
                    double scale = lr * error;
                    for (size_t w_index = 0; w_index < w_count; w_index++) {
                        weights[w_index] += scale * values_[w_index];
                    }

                    weights[w_count] += scale;
                    model->errors[layer][neuron] = error;
                }
            }

            // check error sum each ERROR_VALIDATION_STEP iterations
            int input_it = training_data->meta.size*it + (i + 1);
            if (input_it % ERROR_VALIDATION_STEP == 0) {
                global_error /= ERROR_VALIDATION_STEP;
                DA_APPEND(model->error_hist, ((Error) { .iteration = input_it, .value = global_error }));
                if (global_error < model->training_parameters->tolerance) {
                    goto CLEAN_UP;
                }
                global_error = 0.;
            }
        }
    }

CLEAN_UP:
    model->training = false;
}

typedef struct {
    RNA_Model *model;
    Data *training_data;
} Thread_Args;

internal void* _train_model_async(void *args) {
    Thread_Args *td = (Thread_Args *) args;
    train_model(td->model, td->training_data);
    save_model(td->model);
    return NULL;
}

void train_model_async(RNA_Model *model, Data *training_data) {
    static pthread_t handle;
    static Thread_Args args;
    args.model = model;
    args.training_data = training_data;
    model->training = true;
    pthread_create(&handle, NULL, _train_model_async, (void *)&args);
    pthread_detach(handle);
}

void train_model(RNA_Model *model, Data *training_data) {
    srand(time(NULL));
    struct timespec start, end;

    assert(clock_gettime(CLOCK_MONOTONIC, &start) >= 0);
    _train_model(model, training_data);
    assert(clock_gettime(CLOCK_MONOTONIC, &end) >= 0);

    float start_sec = start.tv_sec + start.tv_nsec/10e9;
    float end_sec = end.tv_sec + end.tv_nsec/10e9;
    float diff_in_secs = end_sec - start_sec;
    printf("Total training time: %.3f secs\n", diff_in_secs);
}

internal void allocate_model(RNA_Model *model) {
    uint32_t layer_cnt = model->layer_count;
    model->values = malloc(sizeof(*model->values) * layer_cnt);
    assert(model->values != NULL);

    model->errors = malloc(sizeof(*model->errors) * layer_cnt);
    assert(model->errors != NULL);

    model->weights = malloc(sizeof(model->weights) * layer_cnt);
    assert(model->weights != NULL);

    model->weights_cnt = malloc(sizeof(*model->weights_cnt) * layer_cnt);
    assert(model->weights_cnt != NULL);

    if (model->neuron_cnt == NULL) {
        model->neuron_cnt = malloc(sizeof(*model->neuron_cnt) * layer_cnt);
        assert(model->neuron_cnt != NULL);
    }
}

internal char *concat_path(char *dir, char *file) {
    assert(file != NULL);
    if (dir == NULL) return file;
    int dir_len = strlen(dir) + 1;
    assert(dir_len < 512 - 1 && "directory name is too big");
    static char buffer[512];
    memcpy(buffer, dir, dir_len);
    buffer[dir_len-1] = '/';
    strncpy(buffer + dir_len, file, 512 - dir_len);
    return buffer;
}

static const char magic[] = "JRNA";
internal bool dump_model(char *out, RNA_Model *model) {
    FILE *f = fopen(out, "wb");
    bool ok = false;
    if (f == NULL) {
        fprintf(stderr, "ERROR: cannot open file %s\n", out);
        goto ERROR;
    }

    if (fwrite(magic, 1, ARR_SIZE(magic) - 1, f) == 0) {
        LOG_WRITE_ERROR(out);
        goto ERROR;
    }

    if (fwrite(&model->layer_count, sizeof(model->layer_count), 1, f) == 0) {
        LOG_WRITE_ERROR(out);
        goto ERROR;
    }

    for (size_t layer = 0; layer < model->layer_count; layer++) {
        uint32_t neurons_cnt = model->neuron_cnt[layer];
        uint32_t weights_cnt = model->weights_cnt[layer];
        uint32_t total_weights = neurons_cnt*(weights_cnt + 1); // + 1 bias

        if (fwrite(&neurons_cnt, sizeof(neurons_cnt), 1, f) == 0) {
            LOG_WRITE_ERROR(out);
            goto ERROR;
        }

        if (fwrite(&weights_cnt, sizeof(weights_cnt), 1, f) == 0) {
            LOG_WRITE_ERROR(out);
            goto ERROR;
        }

        if (fwrite(model->weights[layer], sizeof(double), total_weights, f) != total_weights) {
            LOG_WRITE_ERROR(out);
            goto ERROR;
        }
    }

    ok = true;
ERROR:
    if(f) fclose(f);
    return ok;
}

bool load_model(char *in, RNA_Model *model) {
    FILE *f = fopen(in, "rb");
    bool status = false;
    if (f == NULL) {
        fprintf(stderr, "ERROR: cannot open file %s\n", in);
        goto ERROR;
    }

    static unsigned char magic_buffer[4];
    if (fread(&magic_buffer, sizeof(magic_buffer), 1, f) == 0) {
        LOG_READ_ERROR("magic value", in);
        goto ERROR;
    }

    if (memcmp(magic, magic_buffer, 4) != 0) {
        fprintf(stderr, "ERROR: first 4 bytes of file %s dont match magic constant %*s\n", in, 4, magic);
        goto ERROR;
    }

    if (fread(&model->layer_count, sizeof(model->layer_count), 1, f) == 0) {
        LOG_READ_ERROR("layer_count", in);
        goto ERROR;
    }

    // pre allocate a bunch of fields
    allocate_model(model);
    for (size_t layer = 0; layer < model->layer_count; layer++) {
        if (fread(&model->neuron_cnt[layer], sizeof(model->neuron_cnt[layer]), 1, f) == 0) {
            LOG_READ_ERROR("neuron count", in);
            goto ERROR;
        }

        if (fread(&model->weights_cnt[layer], sizeof(model->weights_cnt[layer]), 1, f) == 0) {
            LOG_READ_ERROR("weights count", in);
            goto ERROR;
        }

        model->values[layer] = malloc(sizeof(*model->values[layer]) * model->neuron_cnt[layer]);
        assert(model->values[layer] != NULL);

        // + 1 for bias
        uint32_t total_weights = (model->weights_cnt[layer] + 1) * model->neuron_cnt[layer];
        model->weights[layer] = malloc(sizeof(double)*total_weights);
        assert(model->weights[layer] != NULL);
        if (fread(model->weights[layer], sizeof(double), total_weights, f) != total_weights) {
            LOG_READ_ERROR("weights", in);
            goto ERROR;
        }
    }

    status = true;
ERROR:
    if(f) fclose(f);
    return status;
}

bool save_model(RNA_Model *model) {
    if (model->training_parameters->output_path == NULL) {
        static char buffer[256];
        sprintf(buffer, "lr_%.4f-tl_%.4f-itrs_%d.model", model->training_parameters->lr, model->training_parameters->tolerance, model->training_parameters->max_iters);
        if (!dump_model(concat_path(model->training_parameters->output_dir_path, buffer), model)) {
            fprintf(stderr, "ERROR: failed to save model\n");
            return false;
        }
    }

    char *path = concat_path(model->training_parameters->output_dir_path, model->training_parameters->output_path);
    if (!dump_model(path, model)) {
        fprintf(stderr, "ERROR: failed to save model\n");
        return false;
    }

    printf("INFO: saved model to file %s\n", path);
    return true;
}

internal double rand_w(float min, float max) {
    return min + (float)rand()/(float)RAND_MAX*(max-min);
}

// Make sure to the the `neuron_cnt` in the model before initialization
void init_model(RNA_Model *model, Data *data) {
    allocate_model(model);
    if (model->training_parameters == NULL) {
        model->training_parameters = &default_parameters;
    }

    for (size_t layer = 0; layer < model->layer_count; layer++) {
        model->values[layer] = malloc(sizeof(*model->values[layer]) * model->neuron_cnt[layer]);
        assert(model->values[layer] != NULL);

        model->errors[layer] = malloc(sizeof(*model->errors[layer]) * model->neuron_cnt[layer]);
        assert(model->errors[layer] != NULL);

        if (layer == 0) {
            model->weights_cnt[layer] = data->meta.cols*data->meta.rows;
        } else {
            model->weights_cnt[layer] = model->neuron_cnt[layer - 1];
        }

        // + 1 for bias
        size_t total_weights = (model->weights_cnt[layer] + 1) * model->neuron_cnt[layer];
        model->weights[layer] = malloc(sizeof(double)*total_weights);
        for (size_t w = 0; w < total_weights; w++) {
            model->weights[layer][w] = rand_w(-0.5, 0.5);
        }
    }
}