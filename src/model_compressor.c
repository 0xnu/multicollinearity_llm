#include "../include/model_compressor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

Model* init_model(size_t num_layers, size_t* layer_sizes) {
    Model* model = malloc(sizeof(Model));
    model->num_layers = num_layers;
    model->layers = malloc(num_layers * sizeof(Layer));

    for (size_t i = 0; i < num_layers; i++) {
        model->layers[i].size = layer_sizes[i];
        model->layers[i].weights = malloc(layer_sizes[i] * sizeof(float));
        // Initialize weights with random values for this example
        for (size_t j = 0; j < layer_sizes[i]; j++) {
            model->layers[i].weights[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }

    return model;
}

void free_model(Model* model) {
    for (size_t i = 0; i < model->num_layers; i++) {
        free(model->layers[i].weights);
    }
    free(model->layers);
    free(model);
}

static float compute_correlation(float* x, float* y, size_t n) {
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    
    for (size_t i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }
    
    float numerator = n * sum_xy - sum_x * sum_y;
    float denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    
    return numerator / denominator;
}

void compress_model(Model* model, float threshold) {
    printf("Starting compression with threshold: %f\n", threshold);
    for (size_t layer = 0; layer < model->num_layers; layer++) {
        size_t original_size = model->layers[layer].size;
        float* weights = model->layers[layer].weights;
        
        printf("Processing layer %zu, original size: %zu\n", layer, original_size);
        
        // Create a mask for weights to keep
        int* keep_mask = calloc(original_size, sizeof(int));
        size_t new_size = 0;
        
        for (size_t i = 0; i < original_size; i++) {
            int keep = 1;
            for (size_t j = 0; j < i; j++) {
                if (keep_mask[j]) {
                    float correlation = fabs(compute_correlation(&weights[i], &weights[j], original_size));
                    if (correlation > threshold) {
                        keep = 0;
                        break;
                    }
                }
            }
            if (keep) {
                keep_mask[i] = 1;
                new_size++;
            }
        }
        
        printf("Layer %zu, new size after compression: %zu\n", layer, new_size);
        
        // Create new compressed weights
        float* new_weights = malloc(new_size * sizeof(float));
        size_t index = 0;
        for (size_t i = 0; i < original_size; i++) {
            if (keep_mask[i]) {
                new_weights[index++] = weights[i];
            }
        }
        
        // Update the layer
        free(weights);
        model->layers[layer].weights = new_weights;
        model->layers[layer].size = new_size;
        
        free(keep_mask);
    }
    printf("Compression completed\n");
}

void print_model(const Model* model) {
    printf("Model structure:\n");
    for (size_t i = 0; i < model->num_layers; i++) {
        printf("Layer %zu: %zu neurons\n", i, model->layers[i].size);
    }
}