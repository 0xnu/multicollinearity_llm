#include "../include/model_compressor.h"
#include <stdio.h>

int main() {
    size_t layer_sizes[] = {1000, 500, 100};
    Model* model = init_model(3, layer_sizes);

    printf("Original model:\n");
    print_model(model);

    float threshold = 0.1;
    printf("\nCompressing model with threshold: %f\n", threshold);
    compress_model(model, threshold);

    printf("\nCompressed model:\n");
    print_model(model);

    free_model(model);
    return 0;
}