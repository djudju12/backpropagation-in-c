#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <raylib.h>
#include <raymath.h>

#include "training.h"

#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))
#define MARK_SIZE 32
#define IMAGE_SIZE 28
#define WINDOW_SIZE IMAGE_SIZE*IMAGE_SIZE
#define PADDING 100
#define CARD_SIZE 60
#define CARD_PADDING 15

static Color pixels[WINDOW_SIZE][WINDOW_SIZE];

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

bool load_model_and_init_test_gui(char *model_path) {
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
"  %s --model <model-file>\n"
"  %s --help\n",
    program_name, program_name);

    printf(
"\nOptions:\n"
"  --help               Prints this message.\n"
"  --model <file>       Input model file.\n");
}

int main(int argc, char **argv) {
    char *program_name = shift(&argc, &argv);
    if (argc == 0) {
        usage(program_name);
        return 1;
    }

    char *mode = shift(&argc, &argv);
    if (strcmp(mode, "--model") == 0) {
        if (argc == 0) {
            fprintf(stderr, "ERROR: missing parameter '--model' value\n");
            usage(program_name);
            return 1;
        }

        char *model_path = shift(&argc, &argv);
        if (!load_model_and_init_test_gui(model_path)) {
            fprintf(stderr, "ERROR: could not initialize GUI");
            return 1;
        }
    } else if (strcmp(mode, "--help") == 0) {
        usage(program_name);
    } else {
        printf("ERROR: invalid parameter %s\n", mode);
        usage(program_name);
        return 1;
    }

    return 0;
}