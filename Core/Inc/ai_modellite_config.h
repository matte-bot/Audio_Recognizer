/*
 * ai_model_config.h
 *
 *  Created on: Oct 12, 2024
 *      Author: User
 */

#ifndef AI_MODELLITE_CONFIG_H
#define AI_MODELLITE_CONFIG_H

#include <stdint.h>
#include "stdlib.h"
#include "ai_platform.h"// Include per i tipi di dati standard

// Definizione della struttura ai_modellite_config
typedef struct {
    ai_buffer activations;       // Buffer per le attivazioni
    ai_buffer weights;           // Buffer per i pesi del modello
    size_t activations_size; 	// Dimensione delle attivazioni
    size_t weights_size;     	// Dimensione dei pesi
}ai_model_config;

#endif // AI_MODELLITE_CONFIG_H
