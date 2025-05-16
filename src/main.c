#define __USE_POSIX199309
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include <raylib.h>
#include <raymath.h>

#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))
#define MARK_SIZE 32
#define IMAGE_SIZE 28
#define WINDOW_SIZE IMAGE_SIZE*IMAGE_SIZE
#define PADDING 100
#define CARD_SIZE 60
#define CARD_PADDING 15

#define LOG_WRITE_ERROR(file) fprintf(stderr, "ERROR: could not write to file %s\n", file)
#define LOG_READ_ERROR(msg, file) fprintf(stderr, "ERROR: could not read %s from file %s\n", msg, file)

static struct {
    double tolerance;
    double lr;
    int max_iters;
    char *output_path;
    char *output_dir_path;
    int threads;
    bool verbose;
} training_parameters = {
    .lr = 0.5,
    .tolerance = 0.02,
    .max_iters = 50,
    .verbose = false
};

// TODO:
// We use pointers directly in the model because its easy to deserialize in this way
// Offcourse, its way more error prone handle pointer arithmetics. A more consice memory
// layout can be considered
typedef struct {
    double **weights;          // weigths of neuron `x` in the layer `y` = (weigths[y] + x*weights_cnt[y])
    uint32_t *weights_cnt;
    uint32_t *neuron_cnt;
    uint32_t layer_count;

    // transient fields
    double **errors;          // layer -> neuron -> error
    double **values;          // layer -> neuron -> values

} RNA_Model;

static Color pixels[WINDOW_SIZE][WINDOW_SIZE];

typedef struct {
    struct {
        uint32_t size;
        uint32_t rows;
        uint32_t cols;
    } meta;

    double *_images;
    uint8_t *labels;
} Data;

void print_image(double *image) {
    static char alphabet[] = { '.', ':', '-', '=', '+', '*', '#', '%', '@' };

    for (size_t y = 0; y < IMAGE_SIZE; y++) {
        for (size_t x = 0; x < IMAGE_SIZE; x++) {
            uint8_t v = image[IMAGE_SIZE*y + x] * 255.0;
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

double *get_neuron_weights(RNA_Model *model, uint32_t layer, uint32_t neuron) {
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

void train_model(RNA_Model *model, Data *training_data) {
    const size_t total_pixels = training_data->meta.rows*training_data->meta.cols;

    double global_error = 0;
    for (int it = 0; it < training_parameters.max_iters; it++) {
        for (size_t i = 0; i < training_data->meta.size; i++) {
            size_t image_index = (i * total_pixels);
            int8_t label = training_data->labels[i];

            double *values = training_data->_images + image_index;
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

            double local_error = 0.;
            const size_t out_layer = model->layer_count - 1;
            for (int out_neuron = 0; out_neuron < (int) model->neuron_cnt[out_layer]; out_neuron++) {
                double desired = out_neuron == label ? 1. : 0.;
                double y = model->values[out_layer][out_neuron];
                double error = (desired - y) * y * (1.0 - y);
                local_error += (desired - y) * (desired - y);
                double *weights = get_neuron_weights(model, out_layer, out_neuron);
                for (size_t w_index = 0; w_index < model->weights_cnt[out_layer]; w_index++) {
                    double delta = training_parameters.lr * error * model->values[out_layer - 1][w_index];
                    weights[w_index] += delta;
                }

                weights[model->weights_cnt[out_layer]] += training_parameters.lr * error;
                model->errors[out_layer][out_neuron] = error;
            }

            global_error += local_error*0.5;

            for (int layer = out_layer - 1; layer >= 0; layer--) {
                for (size_t neuron = 0; neuron < model->neuron_cnt[layer]; neuron++) {
                    double error_sum = 0.0;

                    for (size_t next_neuron = 0; next_neuron < model->neuron_cnt[layer + 1]; next_neuron++) {
                        double error = model->errors[layer + 1][next_neuron];
                        double w = get_neuron_weights(model, layer + 1, next_neuron)[neuron];
                        error_sum += w*error;
                    }

                    double y = model->values[layer][neuron];
                    double error = error_sum * y * (1.0 - y);

                    double *values_ = layer == 0 ? (training_data->_images + image_index) : model->values[layer - 1];
                    double *weights = get_neuron_weights(model, layer, neuron);
                    for (size_t w_index = 0; w_index < model->weights_cnt[layer]; w_index++) {
                        weights[w_index] += training_parameters.lr * error * values_[w_index];
                    }

                    weights[model->weights_cnt[layer]] += training_parameters.lr * error;
                    model->errors[layer][neuron] = error;
                }
            }
        }

        if (global_error/training_data->meta.size < training_parameters.tolerance) {
            break;
        }
    }
}

void print_parameters() {
    // +------------------------+
    // | Training Parameters    |
    // +---------------+--------+
    // | Tolerance     | 0.0200 |
    // | Learning Rate | 0.5000 |
    // | Max Iterations|      1 |
    // | Threads       |      4 |
    // +---------------+--------+
    printf("\n+---------------------+\n");
    printf("| Training Parameters |\n");
    printf("+------------+--------+\n");
    printf("| Tolerance  | %.4f |\n", training_parameters.tolerance);
    printf("| LR         | %.4f |\n", training_parameters.lr);
    printf("| Max Iters  |   %04d |\n", training_parameters.max_iters);
    printf("| N Threads  |     %2d |\n", training_parameters.threads);
    printf("+------------+--------+\n");
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

void allocate_model(RNA_Model *model) {
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

static const char magic[] = "JRNA";
bool dump_model(char *out, RNA_Model *model) {
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

        model->errors[layer] = malloc(sizeof(*model->errors[layer]) * model->neuron_cnt[layer]);
        assert(model->errors[layer] != NULL);

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

char *concat_path(char *dir, char *file) {
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

double rand_w(float min, float max) {
    return min + (float)rand()/(float)RAND_MAX*(max-min);
}

// Make sure to the the `neuron_cnt` in the model before initialization
void init_model(RNA_Model *model, Data *data) {
    allocate_model(model);

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

bool test_model(RNA_Model *model) {
    Data testing_data = {0};
    if (!read_data("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &testing_data)) {
        return false;
    }

    int correct_guesses = 0;
    for (uint32_t i = 0; i < testing_data.meta.size; i++) {
        double *image = testing_data._images + (i * testing_data.meta.cols * testing_data.meta.rows);
        uint8_t label = find_label(model, image);
        if (label == testing_data.labels[i]) correct_guesses++;
#ifdef PRINT_ASCII_IMAGES
        printf("Guessed: %d Actual: %d\n", label, testing_data.labels[i]);
        print_image(image, testing_data.meta.cols);
#endif
    }

    print_results(testing_data.meta.size, correct_guesses);
    return true;
}

bool init_training() {

    srand(time(NULL)); // ATENTION: Make sure to use 1 seed per thread

    static const char *images_file_path = "./data/train-images.idx3-ubyte";
    static const char *labels_file_path = "./data/train-labels.idx1-ubyte";
    Data data = {0};
    if (!read_data(images_file_path, labels_file_path, &data)) {
        return false;
    }

    RNA_Model model = { .neuron_cnt = (uint32_t [3]) {64, 32, 10}, .layer_count = 3 };
    print_parameters();

    struct timespec start, end;
    assert(clock_gettime(CLOCK_MONOTONIC, &start) >= 0);
    init_model(&model, &data);
    train_model(&model, &data);
    assert(clock_gettime(CLOCK_MONOTONIC, &end) >= 0);

    float start_sec = start.tv_sec + start.tv_nsec/10e9;
    float end_sec = end.tv_sec + end.tv_nsec/10e9;
    float diff_in_secs = end_sec - start_sec;
    printf("Total training time: %.3f secs\n", diff_in_secs);

    if (!test_model(&model)) {
        fprintf(stderr, "ERROR: failed to test model\n");
        return false;
    }

    if (training_parameters.output_path == NULL) {
        static char buffer[256];
        sprintf(buffer, "lr_%.4f-tl_%.4f-itrs_%d.model", training_parameters.lr, training_parameters.tolerance, training_parameters.max_iters);
        if (!dump_model(concat_path(training_parameters.output_dir_path, buffer), &model)) {
            fprintf(stderr, "ERROR: failed to save model\n");
            return false;
        }
    } else {
        if (!dump_model(concat_path(training_parameters.output_dir_path, training_parameters.output_path), &model)) {
            fprintf(stderr, "ERROR: failed to save model\n");
            return false;
        }
    }

    return true;
}

bool load_model_and_test(char *model_path) {
    RNA_Model model = {0};
    if (!load_model(model_path, &model)) {
        return false;
    }

    if (!test_model(&model)) {
        fprintf(stderr, "ERROR: failed to test model\n");
        return false;
    }

    return true;
}

void mark_mouse_pos(Color color) {
    Vector2 pos = GetMousePosition();
    int px = (int) (pos.x - MARK_SIZE/2);
    int py = (int) (pos.y - MARK_SIZE/2);

    for (int y = 0; y < MARK_SIZE; y++) {
        int cy = py + y;
        for (int x = 0; x < MARK_SIZE; x++) {
            int cx = px + x;
            if (cy > PADDING && cx > PADDING && cy < (WINDOW_SIZE - PADDING) && cx < (WINDOW_SIZE - PADDING)) {
                pixels[cy][cx] = color;
            }
        }
    }
}

Color * make_screen_pixels(double *image) {
    static Color __pixels[IMAGE_SIZE][IMAGE_SIZE];

    for (size_t y = 0; y < IMAGE_SIZE; y++) {
        for (size_t x = 0; x < IMAGE_SIZE; x++) {
            float pixel = *(image + y*IMAGE_SIZE + x) * 255.0;
            __pixels[y][x].r = pixel;
            __pixels[y][x].g = pixel;
            __pixels[y][x].b = pixel;
            __pixels[y][x].a = 255;
        }
    }


    return (Color *) __pixels;
}

double *make_image() {
    static double buffer[IMAGE_SIZE][IMAGE_SIZE];

    Image user_draw = ImageFromImage(LoadImageFromScreen(), (Rectangle) {
        .width = WINDOW_SIZE - PADDING*2,
        .height = WINDOW_SIZE - PADDING*2,
        .x = PADDING,
        .y = PADDING
    });

    ImageColorGrayscale(&user_draw);

    ImageResizeNN(&user_draw, IMAGE_SIZE, IMAGE_SIZE);
    for (size_t y = 0; y < IMAGE_SIZE; y++) {
        for (size_t x = 0; x < IMAGE_SIZE; x++) {
            Color color = GetImageColor(user_draw, x, y);
            buffer[y][x] = (color.r + color.g + color.b)/255.0;
        }
    }

    return (double*) buffer;
}

void draw_card(int i, int current_image, int total_cards, float xoffset_cards, double *images, Texture2D texture) {
    int image_i = current_image + i - total_cards/2;
    if (image_i >= 0) {
        UpdateTexture(texture, make_screen_pixels(images + (int)(image_i*IMAGE_SIZE*IMAGE_SIZE)));

        DrawTexturePro(
            texture,
            (Rectangle) { .height = IMAGE_SIZE, .width =  IMAGE_SIZE, .x = 0, .y = 0 },
            (Rectangle) { .height = CARD_SIZE, .width = CARD_SIZE, .x = xoffset_cards + (CARD_SIZE + CARD_PADDING)*i, .y = WINDOW_SIZE - PADDING/2 - CARD_SIZE/2 },
            (Vector2) {0},
            0.0,
            WHITE
        );
    }


    DrawRectangleLines(
        xoffset_cards + (CARD_SIZE + CARD_PADDING)*i,
        WINDOW_SIZE - PADDING/2 - CARD_SIZE/2,
        CARD_SIZE,
        CARD_SIZE,
        image_i == current_image ? GREEN : WHITE
    );
}

bool load_model_and_init_gui(char *model_path) {
    RNA_Model model = {0};
    if (!load_model(model_path, &model)) {
        return false;
    }

    Data testing_data = {0};
    if (!read_data("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &testing_data)) {
        return false;
    }

    InitWindow(WINDOW_SIZE, WINDOW_SIZE, "Number Recognition");

    for (size_t y = 0; y < WINDOW_SIZE; y++) {
        for (size_t x = 0; x < WINDOW_SIZE; x++) {
            pixels[y][x] = BLACK;
        }
    }

    size_t total_cards = (WINDOW_SIZE-PADDING*2)/(CARD_SIZE+CARD_PADDING);
    if (total_cards%2==0) total_cards--;
    float xoffset_cards = WINDOW_SIZE/2.0 - ((CARD_SIZE+CARD_PADDING)*total_cards)/2.0 + CARD_PADDING/2.0;

    Image img = GenImageColor(IMAGE_SIZE, IMAGE_SIZE, BLACK);
    Texture2D texture = LoadTextureFromImage(img);
    Texture2D window_texture = LoadTextureFromImage(LoadImageFromScreen());
    Texture2D card_textures[total_cards];
    for (size_t i = 0; i < total_cards; i++) {
        card_textures[i] = LoadTextureFromImage(GenImageColor(IMAGE_SIZE, IMAGE_SIZE, BLACK));
    }

    int8_t guess = find_label(&model, testing_data._images);
    UpdateTexture(texture, make_screen_pixels(testing_data._images));
    size_t image_index = 0;
    bool drawining_mode = false;

    int8_t label = -1;
    static char buffer[32];
    size_t current_image = 0;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_M)) drawining_mode = !drawining_mode;

        if (drawining_mode) {
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                mark_mouse_pos(WHITE);
                UpdateTexture(window_texture, pixels);
            } else if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
                mark_mouse_pos(BLACK);
                UpdateTexture(window_texture, pixels);
            } else if (IsKeyPressed(KEY_SPACE))  {
                double *image = make_image();
                label = find_label(&model, image);
            } else if (IsKeyPressed(KEY_R)) {
                for (size_t y = 0; y < WINDOW_SIZE; y++) {
                    for (size_t x = 0; x < WINDOW_SIZE; x++) {
                        pixels[y][x] = BLACK;
                    }

                    label = -1;
                }

                UpdateTexture(window_texture, pixels);
            }

            if (label < 0) {
                sprintf(buffer, "Guess: <space>");
            } else {
                sprintf(buffer, "Guess: %d", label);
            }

            BeginDrawing();
            ClearBackground(BLACK);

            DrawTexture(window_texture, 0, 0, WHITE);
            DrawText("Press <R> to clear", PADDING, PADDING/2.0, 28, WHITE);
            DrawText(
                buffer,
                WINDOW_SIZE - PADDING - MeasureText(buffer, 28),
                PADDING/2.0,
                28,
                WHITE
            );

            DrawRectangleLines(PADDING, PADDING, WINDOW_SIZE - PADDING*2, WINDOW_SIZE - PADDING*2, WHITE);
            EndDrawing();
        } else {
            if (IsKeyPressed(KEY_RIGHT) && current_image < testing_data.meta.size) {
                current_image++;
                image_index = current_image*IMAGE_SIZE*IMAGE_SIZE;
                guess = find_label(&model, testing_data._images + image_index);
                UpdateTexture(texture, make_screen_pixels(testing_data._images + image_index));
            } else if (IsKeyPressed(KEY_LEFT) && current_image > 0) {
                current_image--;
                image_index = current_image*IMAGE_SIZE*IMAGE_SIZE;
                guess = find_label(&model, testing_data._images + image_index);
                UpdateTexture(texture, make_screen_pixels(testing_data._images + image_index));
            }

            BeginDrawing();
            ClearBackground(BLACK);
            sprintf(buffer, "Actual: %d", testing_data.labels[current_image]);
            DrawText(
                buffer,
                PADDING,
                PADDING/2.0,
                28,
                WHITE
            );

            sprintf(buffer, "Guess: %d", guess);
            DrawText(
                buffer,
                WINDOW_SIZE - PADDING - MeasureText(buffer, 28),
                PADDING/2.0,
                28,
                guess != testing_data.labels[current_image] ? RED : WHITE
            );

            // Selected number
            DrawTexturePro(
                texture,
                (Rectangle) { .height = IMAGE_SIZE, .width =  IMAGE_SIZE, .x = 0, .y = 0 },
                (Rectangle) { .height = WINDOW_SIZE - PADDING*2, .width = WINDOW_SIZE - PADDING*2, .x = PADDING, .y = PADDING },
                (Vector2) {0},
                0.0,
                WHITE
            );

            // Cards
            for (size_t i = 0; i < total_cards/2; i++) {
                draw_card(i, current_image, total_cards, xoffset_cards, testing_data._images, card_textures[i]);
            }

            draw_card(total_cards/2, current_image, total_cards, xoffset_cards, testing_data._images, card_textures[total_cards/2]);

            for (size_t i = (total_cards/2) + 1; i < total_cards; i++) {
                draw_card(i, current_image, total_cards, xoffset_cards, testing_data._images, card_textures[i]);
            }

            DrawRectangleLines(PADDING, PADDING, WINDOW_SIZE - PADDING*2, WINDOW_SIZE - PADDING*2, WHITE);
            EndDrawing();
        }

    }

    CloseWindow();

    return true;
}

char* shift(int *argc, char ***argv) {
    return (*argc)--, *(*argv)++;
}

void usage(char *program_name) {
    printf(
"Usage:\n"
"  %s --train [--out <output-file>] [--max-iters <n>] [--tolerance <value>] [--lr <rate>] [--threads <n>]\n"
"  %s --gui --model <model-file>\n"
"  %s --test --model <model-file>\n"
"  %s --help\n",
    program_name, program_name, program_name, program_name);

    printf(
"\nOptions:\n"
"  --help               Prints this message.\n"
"  --train              Train a new model.\n"
"  --gui                Launch the graphical interface.\n"
"  --test               Run test data on the model.\n"
"  --model <file>       Input model file (required for GUI).\n"
"  --out <file>         Output file for the trained model.\n"
"  --out-dir <file>     Output directory for the trained model.\n"
"  --max-iters <n>      Maximum number of iterations [default: %d].\n"
"  --tolerance <value>  Sets the minimum error required to stop training early [default: %.3f].\n"
"  --lr <rate>          Learning rate [default: %.3f].\n"
"  --threads <n>        Number of parallel threads [default: %d].\n",
    training_parameters.max_iters, training_parameters.tolerance, training_parameters.lr, 4);;
}

int main(int argc, char **argv) {
    char *program_name = shift(&argc, &argv);
    if (argc == 0) {
        usage(program_name);
        return 1;
    }

    char *mode = shift(&argc, &argv);
    if (strcmp(mode, "--train") == 0) {
        training_parameters.threads = sysconf(_SC_NPROCESSORS_ONLN);
        while (argc > 0) {
            char *parameter = shift(&argc, &argv);
            if (strcmp(parameter, "--verbose") == 0) {
                training_parameters.verbose = true;
            } else {
                if (argc == 0) {
                    fprintf(stderr, "ERROR: missing parameter '%s' value\n", parameter);
                    usage(program_name);
                    return 1;
                }

                char *value = shift(&argc, &argv);
                if (strcmp(parameter, "--max-iters") == 0) {
                    training_parameters.max_iters = atoi(value);
                } else if (strcmp(parameter, "--tolerance") == 0) {
                    training_parameters.tolerance = atof(value);
                } else if (strcmp(parameter, "--lr") == 0) {
                    training_parameters.lr = atof(value);
                } else if (strcmp(parameter, "--threads") == 0) {
                    training_parameters.threads = atoi(value);
                } else if (strcmp(parameter, "--out") == 0) {
                    training_parameters.output_path = value;
                } else if (strcmp(parameter, "--out-dir") == 0) {
                    training_parameters.output_dir_path = value;
                }
            }

        }

        if (!init_training()) {
            return 1;
        }
    } else if (strcmp(mode, "--gui") == 0) {
        if (argc == 0 || strcmp(shift(&argc, &argv), "--model") != 0) {
            fprintf(stderr, "ERROR: missing parameter '--model'\n");
            usage(program_name);
            return 1;
        }

        if (argc == 0) {
            fprintf(stderr, "ERROR: missing parameter '--model' value\n");
            usage(program_name);
            return 1;
        }

        char *model_path = shift(&argc, &argv);
        if (!load_model_and_init_gui(model_path)) {
            return 1;
        }
    } else if (strcmp(mode, "--test") == 0) {
        if (argc == 0 || strcmp(shift(&argc, &argv), "--model") != 0) {
            fprintf(stderr, "ERROR: missing parameter '--model'\n");
            usage(program_name);
            return 1;
        }

        if (argc == 0) {
            fprintf(stderr, "ERROR: missing parameter '--model' value\n");
            usage(program_name);
            return 1;
        }

        char *model_path = shift(&argc, &argv);
        if (!load_model_and_test(model_path)) {
            return 1;
        }
    } else if (strcmp(mode, "--help") == 0) {
        usage(program_name);
    } else {
        printf("ERROR: invalid mode %s\n", mode);
        usage(program_name);
        return 1;
    }

    return 0;
}