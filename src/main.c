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

#include <raylib.h>

#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))

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

#define MARK_SIZE 32
#define IMAGE_SIZE 28
#define WINDOW_SIZE IMAGE_SIZE*IMAGE_SIZE
#define PADDING 100

static Color pixels[WINDOW_SIZE][WINDOW_SIZE];

typedef struct {
    double *ws;
    uint32_t wcount;
    uint8_t label;
    float stop_cond;
    int it;
} Perceptron;

typedef struct {
    struct {
        uint32_t size;
        uint32_t rows;
        uint32_t cols;
    } meta;

    double *_images;
    uint8_t *labels;
} Data;

typedef struct {
    Perceptron *ps;
    Data *data;
    pthread_mutex_t *mut;
} Thread_Data;

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

double sum_weights(double *image, Perceptron p) {
    double sum = 0;
    for (uint32_t j = 0; j < p.wcount; j++) {
        sum += p.ws[j] * image[j];
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

int find_label(double *image, Perceptron *ps) {
    double max = 0;
    int8_t label = -1;
    for (size_t i = 0; i < 10; i++) {
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
    static size_t head = 0;
    pthread_mutex_lock(mut);
    if (head >= PERCEPTRON_CNT) {
        pthread_mutex_unlock(mut);
        return NULL;
    }

    size_t my_perceptron = head;
    head += 1;

    pthread_mutex_unlock(mut);
    return &queue[my_perceptron];
}

pthread_mutex_t all_trained_mut = PTHREAD_MUTEX_INITIALIZER;
static int trained = 0;
void report_trained() {
    pthread_mutex_lock(&all_trained_mut);
    trained += 1;
    pthread_mutex_unlock(&all_trained_mut);
}

bool all_trained() {
    assert(trained >= 0 && trained <= 10);
    return trained == 10;
}

void *train_perceptron(void *args) {
    Thread_Data *td = (Thread_Data *)args;
    for (Perceptron *p = unqueue(td->ps, td->mut); p != NULL; p = unqueue(td->ps, td->mut)) {
        for (p->it = 0; p->it < training_parameters.max_iters && p->stop_cond > training_parameters.tolerance; p->it++) {
            float sum_stop_cond = 0;
            for (size_t i = 0; i < td->data->meta.size; i++) {
                size_t total_pixels = td->data->meta.rows*td->data->meta.cols;
                size_t image_index = (i * total_pixels);
                double sum = sum_weights(td->data->_images + image_index, *p);
                double y = generate_output(sum);
                double gout = gradient_output(sum);
                double dy = p->label == td->data->labels[i] ? 1.0 : 0.0;
                double local_error = dy - y;
                sum_stop_cond += fabs(local_error);
                for (uint32_t j = 0; j < total_pixels && j < p->wcount; j++) {
                    double i = td->data->_images[image_index + j];
                    p->ws[j] += training_parameters.lr * local_error * i * gout;
                }

                p->ws[p->wcount] += training_parameters.lr * local_error * gout;
            }

            p->stop_cond = sum_stop_cond / td->data->meta.size;
        }

        report_trained();
    }

    return NULL;
}

static void _print_training_status(Perceptron *ps) {
    // +---------------------+
    // | Training Status     |
    // +----+-------+--------+
    // |  N | iters |  error |
    // +----+-------+--------+
    // |  0 |     1 | 0.0148 |
    // |  1 |     1 | 0.0139 |
    // |  2 |     1 | 0.0291 |
    // |  3 |     1 | 0.0348 |
    // |  4 |     1 | 0.0263 |
    // |  5 |     1 | 0.0396 |
    // |  6 |     1 | 0.0202 |
    // |  7 |     1 | 0.0230 |
    // |  8 |     1 | 0.0571 |
    // |  9 |     1 | 0.0492 |
    // +----+-------+--------+


    printf("+---------------------+\n");
    printf("| Training Status     |\n");
    printf("+----+-------+--------+\n");
    printf("|  N | iters |  error |\n");
    printf("+----+-------+--------+\n");
    for (int i = 0; i < 10; i++) {
        printf("\x1b[2K");
        // TODO: better formatting when it passes 100k
        printf("| %2d | %5d | %.4f |\n", ps[i].label, ps[i].it, ps[i].stop_cond);
    }
    printf("+----+-------+--------+\n");
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

void print_training_status(Perceptron *ps) {
    static const int line_cnt = 25;
    printf("\x1b[2J\x1b[H");
    _print_training_status(ps);
    print_parameters();
    printf("\x1b[%dA", line_cnt);
    while (!all_trained()) {
        printf("\x1b[%dA", line_cnt - 9);
        _print_training_status(ps);
    }

    print_parameters();
    printf("\n");
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

static const char magic[] = "PiPi";
bool dump_perceptrons(char *out, Perceptron *ps) {
    FILE *f = fopen(out, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: cannot open file %s\n", out);
        goto ERROR;
    }

    if (fwrite(magic, 1, ARR_SIZE(magic) - 1, f) == 0) {
        fprintf(stderr, "ERROR: could not write to file %s\n", out);
        goto ERROR;
    }

    for (size_t i = 0; i < 10; i++) {
        if (fwrite(&ps[i].label, sizeof(ps[i].label), 1, f) == 0) {
            fprintf(stderr, "ERROR: could not write to file %s\n", out);
            goto ERROR;
        }

        if (fwrite(ps[i].ws, sizeof(*ps[i].ws), ps[i].wcount + 1, f) == 0) {
            fprintf(stderr, "ERROR: could not write to file %s\n", out);
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
        fprintf(stderr, "ERROR: cannot open file %s\n", in);
        goto ERROR;
    }

    static unsigned char magic_buffer[4];
    if (fread(&magic_buffer, sizeof(magic_buffer), 1, f) == 0) {
        fprintf(stderr, "ERROR: cannot read file %s\n", in);
        goto ERROR;
    }

    assert(memcmp(magic, magic_buffer, 4) == 0 && "not a perceptron file");
    for (size_t i = 0; i < 10; i++) {
        ps[i].wcount = 28*28;
        ps[i].ws = malloc((ps[i].wcount + 1) * sizeof(*ps[i].ws));
        ps[i].ws[ps[i].wcount] = 0;
        ps[i].stop_cond = 1;

        if (fread(&ps[i].label, sizeof(ps[i].label), 1, f) == 0) {
            fprintf(stderr, "ERROR: could not read label in file %s\n", in);
            goto ERROR;
        }

        if (fread(ps[i].ws, sizeof(*ps[i].ws), ps[i].wcount + 1, f) == 0) {
            fprintf(stderr, "ERROR: could not read label in file %s\n", in);
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

bool test_model(Perceptron *ps) {
    Data testing_data = {0};
    if (!read_data("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &testing_data)) {
        return false;
    }

    int correct_guesses = 0;
    for (uint32_t i = 0; i < testing_data.meta.size; i++) {
        double *image = testing_data._images + (i * testing_data.meta.cols * testing_data.meta.rows);
        uint8_t label = find_label(image, ps);
        if (label == testing_data.labels[i]) correct_guesses++;
#ifdef PRINT_ASCII_IMAGES
        printf("Guessed: %d Actual: %d\n", label, testing_data.labels[i]);
        print_image(image);
#endif
    }

    print_results(testing_data.meta.size, correct_guesses);
    return true;
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

bool init_training() {
    static const char *images_file_path = "./data/train-images.idx3-ubyte";
    static const char *labels_file_path = "./data/train-labels.idx1-ubyte";
    Data data = {0};
    read_data(images_file_path, labels_file_path, &data);

    Perceptron ps[10] = { 0 };
    uint32_t wcount = data.meta.rows * data.meta.cols;
    for (size_t i = 0; i < 10; i++) {
        ps[i].wcount = wcount;
        ps[i].ws = calloc(wcount + 1, sizeof(*ps[i].ws));
        ps[i].label = i;
        ps[i].stop_cond = 1;
        ps[i].ws[wcount] = 0;
    }

    static pthread_mutex_t perceptron_mut = PTHREAD_MUTEX_INITIALIZER;
    pthread_t threads[training_parameters.threads];

    Thread_Data td = {
        .data = &data,
        .ps = ps,
        .mut = &perceptron_mut
    };


    struct timespec start, end;
    assert(clock_gettime(CLOCK_MONOTONIC, &start) >= 0);
    for (int i = 0; i < training_parameters.threads; i++) {
        pthread_create(&threads[i], NULL, train_perceptron, (void*)&td);
    }

    if (training_parameters.verbose) {
        print_training_status(ps);
    }

    for (int i = 0; i < training_parameters.threads; i++) {
        pthread_join(threads[i], NULL);
    }

    if (!training_parameters.verbose) {
        _print_training_status(ps);
        print_parameters();
    }

    assert(clock_gettime(CLOCK_MONOTONIC, &end) >= 0);

    float start_sec = start.tv_sec + start.tv_nsec/10e9;
    float end_sec = end.tv_sec + end.tv_nsec/10e9;
    float diff_in_secs = end_sec - start_sec;
    printf("Total training time: %.3f secs\n", diff_in_secs);

    if (!test_model(ps)) {
        fprintf(stderr, "ERROR: failed to test model\n");
        return false;
    }

    if (training_parameters.output_path == NULL) {
        static char buffer[256];
        sprintf(buffer, "lr_%.4f-tl_%.4f-itrs_%d.model", training_parameters.lr, training_parameters.tolerance, training_parameters.max_iters);
        if (!dump_perceptrons(concat_path(training_parameters.output_dir_path, buffer), ps)) {
            fprintf(stderr, "ERROR: failed to save model\n");
            return false;
        }
    } else {
        if (!dump_perceptrons(concat_path(training_parameters.output_dir_path, training_parameters.output_path), ps)) {
            fprintf(stderr, "ERROR: failed to save model\n");
            return false;
        }
    }

    return true;
}

bool load_model_and_test(char *model_path) {
    Perceptron ps[10] = {0};
    if (!load_perceptrons(model_path, ps)) {
        return false;
    }

    if (!test_model(ps)) {
        fprintf(stderr, "ERROR: failed to test model\n");
        return false;
    }

    return true;
}

void mark_mouse_pos(Color color) {
    Vector2 pos = GetMousePosition();
    int px = (int) (pos.x - MARK_SIZE/2);
    int py = (int) (pos.y - MARK_SIZE/2);

    for (size_t y = 0; y < MARK_SIZE; y++) {
        int cy = py + y;
        for (size_t x = 0; x < MARK_SIZE; x++) {
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

bool load_model_and_init_gui(char *model_path) {
    Perceptron ps[10] = {0};
    if (!load_perceptrons(model_path, ps)) {
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

    Image img = GenImageColor(IMAGE_SIZE, IMAGE_SIZE, BLACK);
    Texture2D texture = LoadTextureFromImage(img);
    Texture2D window_texture = LoadTextureFromImage(LoadImageFromScreen());
    int8_t guess = find_label(testing_data._images, ps);
    UpdateTexture(texture, make_screen_pixels(testing_data._images));
    uint32_t image_index = 0;
    bool drawining_mode = false;

    int8_t label = -1;
    static char buffer[32];
    uint32_t current_image = 0;
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
                label = find_label(image, ps);
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
                guess = find_label(testing_data._images + image_index, ps);
                UpdateTexture(texture, make_screen_pixels(testing_data._images + image_index));
            } else if (IsKeyPressed(KEY_LEFT) && current_image > 0) {
                current_image--;
                image_index = current_image*IMAGE_SIZE*IMAGE_SIZE;
                guess = find_label(testing_data._images + image_index, ps);
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