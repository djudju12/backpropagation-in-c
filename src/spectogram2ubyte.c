#include <stdio.h>
#include <stdbool.h>
#include <dirent.h>
#include <errno.h>
#include <assert.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))
#define IMAGE_SIZE 128

#define DA_APPEND(da, v) do {                                      \
    if ((da).capacity == 0 || (da).items == NULL) {                \
        (da).capacity = 32;                                        \
        (da).items = malloc((da).capacity*sizeof(v));              \
        assert((da).items != NULL);                                \
    } else if ((da).count >= (da).capacity) {                      \
        (da).capacity *= 2;                                        \
        (da).items = realloc((da).items, (da).capacity*sizeof(v)); \
        assert((da).items != NULL);                                \
    }                                                              \
    (da).items[(da).count++] = v;                                  \
} while (0)

typedef struct {
    int *items;
    size_t count;
    size_t capacity;
} Ids;

char* shift(int *argc, char ***argv) {
    return (*argc)--, *(*argv)++;
}

char *concat_path(char *dir, char *file) {
    assert(file != NULL);
    if (dir == NULL) return (char *) file;
    int dir_len = strlen(dir) + 1;
    assert(dir_len < 512 - 1 && "directory name is too big");
    static char buffer[512];
    memcpy(buffer, dir, dir_len);
    buffer[dir_len-1] = '/';
    strncpy(buffer + dir_len, file, 512 - dir_len);
    return buffer;
}

int32_t count_files(DIR *dir, char *path, char *ext) {
    struct dirent *entry;
    int32_t count = 0;
    errno = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            char *file = concat_path(path, entry->d_name);
            if (strcmp(strrchr(file, '.'), ext) == 0) {
                count++;
            }
        }
    }

    if (errno != 0) {
        fprintf(stderr, "ERROR: something went wrong reading %s. %s", path, strerror(errno));
        return -1;
    }

    rewinddir(dir);
    return count;
}

bool write_ubyte_header(FILE *file, int32_t size) {
    static int32_t header[4];
    header[0] = __builtin_bswap32(2051);
    header[1] = __builtin_bswap32(size);
    header[2] = __builtin_bswap32(IMAGE_SIZE);
    header[3] = __builtin_bswap32(IMAGE_SIZE);

    size_t items_written = fwrite(header, sizeof(header[0]), ARR_SIZE(header), file);
    if (items_written != ARR_SIZE(header)) {
        fprintf(stderr, "ERROR: could not write header to file. Expected writes %ld, actual %ld\n", items_written, ARR_SIZE(header));
        return false;
    }

    return true;
}

int extract_id(char *file) {
    static char buffer[32];
    char *base = strrchr(file, '/');
    base += 1;

    size_t i = 0;
    for (; i < 31 && *base != '.'; i++, base++) {
        buffer[i] = *base;
    }

    buffer[i] = '\0';
    return atoi(buffer);
}

bool create_images_ubyte(DIR *dir, char *dir_path, int32_t count, char *out) {
    FILE *file = fopen(out, "wb");
    assert(file != NULL);

    bool status = true;
    if (!write_ubyte_header(file, count)) {
        status = false;
        goto CLEAN_UP;
    }

    struct dirent *entry;
    for (int32_t i = 0; i < count;) {
        entry = readdir(dir);
        if (entry == NULL) {
            fprintf(stderr, "ERROR: unexpected end of directory\n");
            return false;
        }

        if (entry->d_type != DT_REG) continue;
        char *file_path = concat_path(dir_path, entry->d_name);

        if (strcmp(strrchr(file_path, '.'), ".png") != 0) continue;

        i++;
        int w, h;
        unsigned char* image = stbi_load(file_path, &w, &h, NULL, STBI_grey);

        if (image == NULL) {
            fprintf(stderr, "ERROR: could not load image to %s\n", file_path);
            return false;
        }

        if (w != IMAGE_SIZE || h != IMAGE_SIZE) {
            fprintf(stderr, "ERROR: image %s dont have correct dimensions. Expected (%dx%d), actual (%dx%d)\n", file_path, IMAGE_SIZE, IMAGE_SIZE, w, h);
            return false;
        }

        size_t size = w*h; // 1 channel
        if (fwrite(image, sizeof(*image), size, file) != size) {
            fprintf(stderr, "ERROR: could not write all data to %s\n", file_path);
            return false;
        }

        printf("%d, %s\n", extract_id(file_path), out);
    }

CLEAN_UP:
    fclose(file);
    return status;
}

int main(int argc, char **argv) {
    shift(&argc, &argv);
    char *path = shift(&argc, &argv);
    if (path == NULL) {
        fprintf(stderr, "ERROR: path to dir in needed\n");
        return 1;
    }

    DIR *dir = opendir(path);
    if (dir == NULL) {
        fprintf(stderr, "ERROR: could not open dir %s. %s", path, strerror(errno));
        return 1;
    }

    int32_t count = count_files(dir, path, ".png");
    assert(count >= 0);

    int32_t training_count = count*0.80;
    int32_t testing_count = count - training_count;

    static char name_buffer[126];
    sprintf(name_buffer, "training-%d.ubyte", training_count);
    if (!create_images_ubyte(dir, path, training_count, name_buffer)) {
        printf("ERROR: could not write images\n");
        return 1;
    }

    sprintf(name_buffer, "testing-%d.ubyte", testing_count);
    if (!create_images_ubyte(dir, path, testing_count, name_buffer)) {
        printf("ERROR: could not write images\n");
        return 1;
    }

    return 0;
}
