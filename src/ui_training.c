#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <raylib.h>
#include <raymath.h>

#include "resources/courier_prime.c"
#include "training.h"

#define SCREEN_WIDTH 510
#define SCREEN_HEIGHT 350
#define CHART_STEP_CNT 5
#define CHART_STEP_LEN 8
#define CHART_STEP_PAD 3
#define CHART_LINE_THICKNESS 2.0
#define CHART_FONT_SIZE 18

typedef struct {
    Vector2 *items;
    size_t capacity;
    size_t count;
} Data_Points;

typedef struct {
    double max_y, max_x;
    int width, height;
    Data_Points points;
} Chart;

Font font;

Vector2 scale_vector(Chart chart, Vector2 dp_pos) {
    return (Vector2) {
        .x = (dp_pos.x * chart.width) / chart.max_x,
        .y = (-1*dp_pos.y * chart.height) / chart.max_y
    };
}

void chart_draw(int x, int y, Chart chart) {
    static char label_buffer[64];
    Vector2 origin = { .x = x, .y = y + chart.height };
    Vector2 last;
    for (size_t i = 0; i < chart.points.count; i++) {
        Vector2 dp_pos = chart.points.items[i];
        Vector2 chart_dp_pos = Vector2Add(scale_vector(chart, dp_pos), origin);
        if (dp_pos.y > 0) {
            if (i != 0) {
                DrawLineEx(last, chart_dp_pos, CHART_LINE_THICKNESS, BLUE);
            }
        }

        last = chart_dp_pos;
    }

    Rectangle chart_border = {
        .height = chart.height,
        .width = chart.width,
        .x = x, .y = y
    };

    DrawRectangleLinesEx(chart_border, CHART_LINE_THICKNESS, BLACK);
    for (int y = 0; y <= CHART_STEP_CNT; y++) {
        Vector2 pos = (Vector2) {
            .x = 0,
            .y = y*(chart.max_y / (double) CHART_STEP_CNT)
        };

        Vector2 scaled_pos = Vector2Add(scale_vector(chart, pos), origin);
        scaled_pos.y += CHART_LINE_THICKNESS/2;
        DrawLineEx(
            scaled_pos,
            Vector2Subtract(scaled_pos, (Vector2) {.x = CHART_STEP_LEN, .y = 0}),
            CHART_LINE_THICKNESS, BLACK
        );

        snprintf(label_buffer, 64, "%.2f", pos.y);
        Vector2 text_size = MeasureTextEx(font, label_buffer, CHART_FONT_SIZE, 1.0);
        scaled_pos.x = origin.x - CHART_STEP_LEN - CHART_STEP_PAD - text_size.x;
        scaled_pos.y -= text_size.y/2.0;
        DrawTextEx(font, label_buffer, scaled_pos, CHART_FONT_SIZE, 1.0, BLACK);
    }

    int step_x = CHART_STEP_CNT > chart.max_x ? chart.max_x : CHART_STEP_CNT;
    for (int x = 0; x <= step_x; x++) {
        Vector2 pos = (Vector2) {
            .x = x*(chart.max_x / (double) step_x),
            .y = 0
        };

        Vector2 scaled_pos = Vector2Add(scale_vector(chart, pos), origin);
        scaled_pos.x -= CHART_LINE_THICKNESS/2;
        DrawLineEx(
            scaled_pos,
            Vector2Add(scaled_pos, (Vector2) {.x = 0, .y = CHART_STEP_LEN}),
            CHART_LINE_THICKNESS, BLACK
        );

        snprintf(label_buffer, 64, "%d", (int) pos.x);
        Vector2 text_size = MeasureTextEx(font, label_buffer, CHART_FONT_SIZE, 1.0);
        scaled_pos.x -= text_size.x/2.0;
        scaled_pos.y += text_size.y/2.0 + CHART_STEP_PAD;
        DrawTextEx(font, label_buffer, scaled_pos, CHART_FONT_SIZE, 1.0, BLACK);
    }
}

void chart_add_dp(Chart *chart, Vector2 dp) {
    DA_APPEND(chart->points, dp);
    if (dp.x > chart->max_x) chart->max_x = dp.x + dp.x*0.2;
    if (dp.y > chart->max_y) chart->max_y = dp.y + dp.y*0.2;
}

void print_results(int total, int correct) {
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
    }

    print_results(testing_data.meta.size, correct_guesses);
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

char* shift(int *argc, char ***argv) {
    return (*argc)--, *(*argv)++;
}

void usage(char *program_name) {
    RNA_Parameters default_parameters = get_default_parameters();
    printf(
"Usage:\n"
"  %s --train [--out <output-file>] [--max-iters <n>] [--tolerance <value>] [--lr <rate>]\n"
"  %s --test --model <model-file>\n"
"  %s --help\n",
    program_name, program_name, program_name);

    printf(
"\nOptions:\n"
"  --help               Prints this message.\n"
"  --train              Train a new model.\n"
"  --test               Run test data on the model.\n"
"  --model <file>       Input model file (required for GUI).\n"
"  --out <file>         Output file for the trained model.\n"
"  --out-dir <file>     Output directory for the trained model.\n"
"  --max-iters <n>      Maximum number of iterations [default: %d].\n"
"  --tolerance <value>  Sets the minimum error required to stop training early [default: %.3f].\n"
"  --lr <rate>          Learning rate [default: %.3f].\n",
    default_parameters.max_iters, default_parameters.tolerance, default_parameters.lr);
}

bool init_training(RNA_Parameters *training_parameters) {
    srand(time(NULL));
    static const char *images_file_path = "./data/train-images.idx3-ubyte";
    static const char *labels_file_path = "./data/train-labels.idx1-ubyte";

    Data data = {0};
    if (!read_data(images_file_path, labels_file_path, &data)) {
        return false;
    }

    RNA_Model model = { .neuron_cnt = (uint32_t []) {64, 32, 10}, .layer_count = 3, .training_parameters = training_parameters };

    init_model(&model, &data);
    train_model_async(&model, &data);

    SetTraceLogLevel(LOG_ERROR);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Training");

    Chart chart = {0};
    font = LoadFont_CourierPrimeRegular();
    chart.width = 350;
    chart.height = 250;
    chart.max_y = 1.0;

    while (!WindowShouldClose()) {
        size_t new_data_count = model.error_hist.count - chart.points.count;
        for (size_t i = 0; i < new_data_count; i++) {
            Error new = model.error_hist.items[chart.points.count];
            chart_add_dp(&chart, (Vector2) { .x = (double) new.iteration, .y = new.value });
        }

        BeginDrawing();

        if (model.error_hist.count > 0) {
            chart_draw(SCREEN_WIDTH/2 - chart.width/2, 30, chart);
        }

        if (!model.training && IsKeyPressed(KEY_T) && !test_model(&model)) {
            printf("ERROR: could not test model\n");
        }

        if (!model.training && IsKeyPressed(KEY_SPACE)) {
            chart.points.count = 0;
            chart.max_x = 0;
            train_model_async(&model, &data);
        }

        ClearBackground((Color){.r = 220, .g = 220, .b = 220, .a = 255});
        EndDrawing();
    }

    CloseWindow();

    return true;
}


int main(int argc, char **argv) {
    char *program_name = shift(&argc, &argv);
    if (argc == 0) {
        usage(program_name);
        return 1;
    }

    char *parameter = shift(&argc, &argv);
    if (strcmp(parameter, "--train") == 0) {
        RNA_Parameters training_parameters = get_default_parameters();
        while (argc > 0) {
            parameter = shift(&argc, &argv);
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
            } else if (strcmp(parameter, "--out") == 0) {
                training_parameters.output_path = value;
            } else if (strcmp(parameter, "--out-dir") == 0) {
                training_parameters.output_dir_path = value;
            } else if (strcmp(parameter, "--config") == 0) {
                training_parameters.config_path = value;
            } else {
                fprintf(stderr, "WARNING: ignoring unknow parameter %s\n", parameter);
            }
        }

        if (!init_training(&training_parameters)) {
            return 1;
        }

    } else if (strcmp(parameter, "--test") == 0) {
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
    } else if (strcmp(parameter, "--help") == 0) {
        usage(program_name);
    } else {
        printf("ERROR: invalid parameter %s\n", parameter);
        usage(program_name);
        return 1;
    }

}