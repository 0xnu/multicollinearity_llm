#include "../include/model_compressor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// Helper function to compute the mean of an array
static float compute_mean(const float* arr, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum / n;
}

// Helper function to compute the standard deviation of an array
static float compute_std(const float* arr, size_t n) {
    float mean = compute_mean(arr, n);
    float sum_squared_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = arr[i] - mean;
        sum_squared_diff += diff * diff;
    }
    return sqrtf(sum_squared_diff / n);
}

// Magnitude-based pruning
void magnitude_pruning(Layer* layer, float threshold) {
    float* weights = layer->weights;
    size_t size = layer->size;
    
    // Compute the threshold value
    float abs_threshold = fabsf(threshold);
    
    // Count the number of weights to keep
    size_t new_size = 0;
    for (size_t i = 0; i < size; i++) {
        if (fabsf(weights[i]) > abs_threshold) {
            new_size++;
        }
    }
    
    // Allocate new weights array
    float* new_weights = malloc(new_size * sizeof(float));
    
    // Copy weights that exceed the threshold
    size_t j = 0;
    for (size_t i = 0; i < size; i++) {
        if (fabsf(weights[i]) > abs_threshold) {
            new_weights[j++] = weights[i];
        }
    }
    
    // Update the layer
    free(weights);
    layer->weights = new_weights;
    layer->size = new_size;
}

// Percentage-based pruning
void percentage_pruning(Layer* layer, float percentage) {
    float* weights = layer->weights;
    size_t size = layer->size;
    
    // Compute the number of weights to keep
    size_t keep_count = (size_t)(size * (1.0f - percentage));
    
    // Create an array of indices
    size_t* indices = malloc(size * sizeof(size_t));
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }
    
    // Sort indices based on absolute weight values (descending order)
    for (size_t i = 0; i < size - 1; i++) {
        for (size_t j = 0; j < size - i - 1; j++) {
            if (fabsf(weights[indices[j]]) < fabsf(weights[indices[j + 1]])) {
                size_t temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
    
    // Allocate new weights array
    float* new_weights = malloc(keep_count * sizeof(float));
    
    // Copy the top 'keep_count' weights
    for (size_t i = 0; i < keep_count; i++) {
        new_weights[i] = weights[indices[i]];
    }
    
    // Update the layer
    free(weights);
    free(indices);
    layer->weights = new_weights;
    layer->size = keep_count;
}

// Random pruning
void random_pruning(Layer* layer, float percentage) {
    float* weights = layer->weights;
    size_t size = layer->size;
    
    // Compute the number of weights to keep
    size_t keep_count = (size_t)(size * (1.0f - percentage));
    
    // Create a mask for weights to keep
    int* keep_mask = calloc(size, sizeof(int));
    
    // Randomly select weights to keep
    for (size_t i = 0; i < keep_count; i++) {
        size_t index;
        do {
            index = rand() % size;
        } while (keep_mask[index]);
        keep_mask[index] = 1;
    }
    
    // Allocate new weights array
    float* new_weights = malloc(keep_count * sizeof(float));
    
    // Copy selected weights
    size_t j = 0;
    for (size_t i = 0; i < size; i++) {
        if (keep_mask[i]) {
            new_weights[j++] = weights[i];
        }
    }
    
    // Update the layer
    free(weights);
    free(keep_mask);
    layer->weights = new_weights;
    layer->size = keep_count;
}

// Variance-based pruning
void variance_pruning(Layer* layer, float threshold) {
    float* weights = layer->weights;
    size_t size = layer->size;
    
    // Compute the standard deviation of weights
    float std_dev = compute_std(weights, size);
    float variance = std_dev * std_dev;
    
    // Count the number of weights to keep
    size_t new_size = 0;
    for (size_t i = 0; i < size; i++) {
        if (fabsf(weights[i]) > (threshold * std_dev)) {
            new_size++;
        }
    }
    
    // Allocate new weights array
    float* new_weights = malloc(new_size * sizeof(float));
    
    // Copy weights that exceed the variance threshold
    size_t j = 0;
    for (size_t i = 0; i < size; i++) {
        if (fabsf(weights[i]) > (threshold * std_dev)) {
            new_weights[j++] = weights[i];
        }
    }
    
    // Update the layer
    free(weights);
    layer->weights = new_weights;
    layer->size = new_size;
}

// L1-norm pruning
void l1_norm_pruning(Layer* layer, float threshold) {
    float* weights = layer->weights;
    size_t size = layer->size;
    
    // Compute L1 norm
    float l1_norm = 0.0f;
    for (size_t i = 0; i < size; i++) {
        l1_norm += fabsf(weights[i]);
    }
    
    // Compute the threshold value
    float abs_threshold = threshold * l1_norm / size;
    
    // Count the number of weights to keep
    size_t new_size = 0;
    for (size_t i = 0; i < size; i++) {
        if (fabsf(weights[i]) > abs_threshold) {
            new_size++;
        }
    }
    
    // Allocate new weights array
    float* new_weights = malloc(new_size * sizeof(float));
    
    // Copy weights that exceed the threshold
    size_t j = 0;
    for (size_t i = 0; i < size; i++) {
        if (fabsf(weights[i]) > abs_threshold) {
            new_weights[j++] = weights[i];
        }
    }
    
    // Update the layer
    free(weights);
    layer->weights = new_weights;
    layer->size = new_size;
}

// Gradient-based pruning (simulated for this example)
void gradient_based_pruning(Layer* layer, float threshold) {
    float* weights = layer->weights;
    size_t size = layer->size;
    
    // Simulate gradients (in a real scenario, these would come from backpropagation)
    float* gradients = malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
        gradients[i] = ((float)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
    }
    
    // Compute the product of weights and gradients
    float* importance = malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
        importance[i] = fabsf(weights[i] * gradients[i]);
    }
    
    // Count the number of weights to keep
    size_t new_size = 0;
    for (size_t i = 0; i < size; i++) {
        if (importance[i] > threshold) {
            new_size++;
        }
    }
    
    // Allocate new weights array
    float* new_weights = malloc(new_size * sizeof(float));
    
    // Copy weights that exceed the importance threshold
    size_t j = 0;
    for (size_t i = 0; i < size; i++) {
        if (importance[i] > threshold) {
            new_weights[j++] = weights[i];
        }
    }
    
    // Update the layer
    free(weights);
    free(gradients);
    free(importance);
    layer->weights = new_weights;
    layer->size = new_size;
}

// Entropy-based pruning
void entropy_based_pruning(Layer* layer, float threshold) {
    float* weights = layer->weights;
    size_t size = layer->size;
    
    // Compute the entropy of weights
    float entropy = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float p = fabsf(weights[i]) / size;
        if (p > 0) {
            entropy -= p * log2f(p);
        }
    }
    
    // Normalize entropy
    float max_entropy = log2f(size);
    float normalized_entropy = entropy / max_entropy;
    
    // Count the number of weights to keep based on entropy
    size_t new_size = (size_t)(size * (1.0f - normalized_entropy * threshold));
    
    // Sort weights by absolute value
    float* sorted_weights = malloc(size * sizeof(float));
    memcpy(sorted_weights, weights, size * sizeof(float));
    for (size_t i = 0; i < size - 1; i++) {
        for (size_t j = 0; j < size - i - 1; j++) {
            if (fabsf(sorted_weights[j]) < fabsf(sorted_weights[j + 1])) {
                float temp = sorted_weights[j];
                sorted_weights[j] = sorted_weights[j + 1];
                sorted_weights[j + 1] = temp;
            }
        }
    }
    
    // Determine the threshold value
    float prune_threshold = fabsf(sorted_weights[new_size - 1]);
    
    // Allocate new weights array
    float* new_weights = malloc(new_size * sizeof(float));
    
    // Copy weights that exceed the threshold
    size_t j = 0;
    for (size_t i = 0; i < size; i++) {
        if (fabsf(weights[i]) >= prune_threshold) {
            new_weights[j++] = weights[i];
        }
    }
    
    // Update the layer
    free(weights);
    free(sorted_weights);
    layer->weights = new_weights;
    layer->size = new_size;
}

// Main pruning function that applies the specified pruning method
void prune_layer(Layer* layer, PruningMethod method, float threshold) {
    switch (method) {
        case MAGNITUDE_PRUNING:
            magnitude_pruning(layer, threshold);
            break;
        case PERCENTAGE_PRUNING:
            percentage_pruning(layer, threshold);
            break;
        case RANDOM_PRUNING:
            random_pruning(layer, threshold);
            break;
        case VARIANCE_PRUNING:
            variance_pruning(layer, threshold);
            break;
        case L1_NORM_PRUNING:
            l1_norm_pruning(layer, threshold);
            break;
        case GRADIENT_BASED_PRUNING:
            gradient_based_pruning(layer, threshold);
            break;
        case ENTROPY_BASED_PRUNING:
            entropy_based_pruning(layer, threshold);
            break;
        default:
            fprintf(stderr, "Unknown pruning method\n");
            break;
    }
}

// Prune the entire model using the specified method
void prune_model(Model* model, PruningMethod method, float threshold) {
    for (size_t i = 0; i < model->num_layers; i++) {
        prune_layer(&model->layers[i], method, threshold);
    }
}