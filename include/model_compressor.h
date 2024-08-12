#ifndef MODEL_COMPRESSOR_H
#define MODEL_COMPRESSOR_H

#include <stddef.h>

typedef struct {
    float* weights;
    size_t size;
} Layer;

typedef struct {
    Layer* layers;
    size_t num_layers;
} Model;

typedef enum {
    MAGNITUDE_PRUNING,
    PERCENTAGE_PRUNING,
    RANDOM_PRUNING,
    VARIANCE_PRUNING,
    L1_NORM_PRUNING,
    GRADIENT_BASED_PRUNING,
    ENTROPY_BASED_PRUNING
} PruningMethod;

// Initialize a model
Model* init_model(size_t num_layers, size_t* layer_sizes);

// Free model memory
void free_model(Model* model);

// Compress model using correlation-based pruning
void compress_model(Model* model, float threshold);

// Prune a single layer using the specified method
void prune_layer(Layer* layer, PruningMethod method, float threshold);

// Prune the entire model using the specified method
void prune_model(Model* model, PruningMethod method, float threshold);

// Helper function to print model details
void print_model(const Model* model);

#endif // MODEL_COMPRESSOR_H