/**
  ******************************************************************************
  * @file    modellite_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2024-10-13T14:50:07+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef MODELLITE_DATA_PARAMS_H
#define MODELLITE_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_MODELLITE_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_modellite_data_weights_params[1]))
*/

#define AI_MODELLITE_DATA_CONFIG               (NULL)


#define AI_MODELLITE_DATA_ACTIVATIONS_SIZES \
  { 3868, }
#define AI_MODELLITE_DATA_ACTIVATIONS_SIZE     (3868)
#define AI_MODELLITE_DATA_ACTIVATIONS_COUNT    (1)
#define AI_MODELLITE_DATA_ACTIVATION_1_SIZE    (3868)



#define AI_MODELLITE_DATA_WEIGHTS_SIZES \
  { 340028, }
#define AI_MODELLITE_DATA_WEIGHTS_SIZE         (340028)
#define AI_MODELLITE_DATA_WEIGHTS_COUNT        (1)
#define AI_MODELLITE_DATA_WEIGHT_1_SIZE        (340028)



#define AI_MODELLITE_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_modellite_activations_table[1])

extern ai_handle g_modellite_activations_table[1 + 2];



#define AI_MODELLITE_DATA_WEIGHTS_TABLE_GET() \
  (&g_modellite_weights_table[1])

extern ai_handle g_modellite_weights_table[1 + 2];


#endif    /* MODELLITE_DATA_PARAMS_H */
