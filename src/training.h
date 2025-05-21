#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

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
    size_t iteration;
    double value;
} Error;

typedef struct {
    Error *items;
    size_t count;
    size_t capacity;
} Error_Hist;

typedef struct {
    double tolerance;
    double lr;
    int max_iters;
    char *output_path;
    char *output_dir_path;
    char *config_path;
} RNA_Parameters;

typedef struct {
    double **weights;          // weigths of neuron `x` in the layer `y` = (weigths[y] + x*weights_cnt[y])
    uint32_t *weights_cnt;
    uint32_t *neuron_cnt;
    uint32_t layer_count;

    // transient fields
    double **errors;          // layer -> neuron -> error
    double **values;          // layer -> neuron -> values
    RNA_Parameters *training_parameters;
    bool training;
    Error_Hist error_hist;
    int epoch;
} RNA_Model;

typedef struct {
    struct {
        uint32_t size;
        uint32_t rows;
        uint32_t cols;
    } meta;

    double *_images;
    uint8_t *labels;
} Data;

bool load_model(char *in, RNA_Model *model); // Load model from file
bool save_model(RNA_Model *model); // Saves model to a file
void init_model(RNA_Model *model, Data *data); // Initilize model (allocation and stuff)
void train_model(RNA_Model *model, Data *training_data); // Train model using the training data (./data/train-*.ubyte)
void train_model_async(RNA_Model *model, Data *training_data);
bool test_model(RNA_Model *model); // Test model using the testing data (./data/tk10k-*.ubyte)
int find_label(RNA_Model *model, double *image); // Find label (0..9) of given image
RNA_Parameters get_default_parameters(void); // Get parameters used in `init_model` when model.training_parameters == NULL
bool read_data(const char *images_file_path, const char *labels_file_path, Data *data);