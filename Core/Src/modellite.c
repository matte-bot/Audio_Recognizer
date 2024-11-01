/**
  ******************************************************************************
  * @file    modellite.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2024-10-13T14:50:07+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "modellite.h"
#include "modellite_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_modellite
 
#undef AI_MODELLITE_MODEL_SIGNATURE
#define AI_MODELLITE_MODEL_SIGNATURE     "0xd93eaca3690414661d1db0c47c312870"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2024-10-13T14:50:07+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_MODELLITE_N_BATCHES
#define AI_MODELLITE_N_BATCHES         (1)

static ai_ptr g_modellite_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_modellite_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input_layer_20_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 455, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gemm_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  nl_1_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  gemm_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  nl_2_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gemm_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  nl_3_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  nl_5_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 15, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  gemm_1_weights_array, AI_ARRAY_FORMAT_LUT8_FLOAT,
  NULL, NULL, 232960, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  gemm_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  gemm_2_weights_array, AI_ARRAY_FORMAT_LUT8_FLOAT,
  NULL, NULL, 65536, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  gemm_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  gemm_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  gemm_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 960, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 15, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  gemm_2_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &gemm_2_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_weights, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 64, 15, 1, 1), AI_STRIDE_INIT(4, 4, 256, 3840, 3840),
  1, &gemm_4_weights_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  nl_2_nl_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &nl_2_nl_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 15, 1, 1), AI_STRIDE_INIT(4, 4, 4, 60, 60),
  1, &gemm_4_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  gemm_3_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_3_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  nl_3_nl_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &nl_3_nl_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 15, 1, 1), AI_STRIDE_INIT(4, 4, 4, 60, 60),
  1, &gemm_4_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  nl_5_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 15, 1, 1), AI_STRIDE_INIT(4, 4, 4, 60, 60),
  1, &nl_5_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  gemm_3_weights, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 128, 64, 1, 1), AI_STRIDE_INIT(4, 4, 512, 32768, 32768),
  1, &gemm_3_weights_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  gemm_1_weights, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 455, 512, 1, 1), AI_STRIDE_INIT(4, 1, 455, 232960, 232960),
  1, &gemm_1_weights_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  gemm_1_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2048, 2048),
  1, &gemm_1_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  gemm_2_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 512, 128, 1, 1), AI_STRIDE_INIT(4, 1, 512, 65536, 65536),
  1, &gemm_2_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  gemm_2_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &gemm_2_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_layer_20_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 13, 1, 35), AI_STRIDE_INIT(4, 4, 4, 52, 52),
  1, &serving_default_input_layer_20_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_layer_20_output0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 455, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1820, 1820),
  1, &serving_default_input_layer_20_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  gemm_1_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2048, 2048),
  1, &gemm_1_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  gemm_3_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_3_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  nl_1_nl_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2048, 2048),
  1, &nl_1_nl_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_5_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_sm,
  &nl_5_chain,
  NULL, &nl_5_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_3_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_4_weights, &gemm_4_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_4_layer, 4,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_4_chain,
  NULL, &nl_5_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_3_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_3_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_3_nl_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_3_nl_chain,
  NULL, &gemm_4_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_2_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_3_weights, &gemm_3_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_3_layer, 3,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_3_chain,
  NULL, &nl_3_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_2_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_2_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_2_nl_layer, 2,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_2_nl_chain,
  NULL, &gemm_3_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_1_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_2_weights, &gemm_2_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_2_layer, 2,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_2_chain,
  NULL, &nl_2_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_1_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_1_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_1_nl_layer, 1,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_1_nl_chain,
  NULL, &gemm_2_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input_layer_20_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_1_weights, &gemm_1_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_1_layer, 1,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_1_chain,
  NULL, &nl_1_nl_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 340028, 1, 1),
    340028, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 3868, 1, 1),
    3868, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODELLITE_IN_NUM, &serving_default_input_layer_20_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODELLITE_OUT_NUM, &nl_5_output),
  &gemm_1_layer, 0x10f4ec69, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 340028, 1, 1),
      340028, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 3868, 1, 1),
      3868, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODELLITE_IN_NUM, &serving_default_input_layer_20_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_MODELLITE_OUT_NUM, &nl_5_output),
  &gemm_1_layer, 0x10f4ec69, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool modellite_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_modellite_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_input_layer_20_output_array.data = AI_PTR(g_modellite_activations_map[0] + 0);
    serving_default_input_layer_20_output_array.data_start = AI_PTR(g_modellite_activations_map[0] + 0);
    gemm_1_output_array.data = AI_PTR(g_modellite_activations_map[0] + 1820);
    gemm_1_output_array.data_start = AI_PTR(g_modellite_activations_map[0] + 1820);
    nl_1_nl_output_array.data = AI_PTR(g_modellite_activations_map[0] + 1820);
    nl_1_nl_output_array.data_start = AI_PTR(g_modellite_activations_map[0] + 1820);
    gemm_2_output_array.data = AI_PTR(g_modellite_activations_map[0] + 0);
    gemm_2_output_array.data_start = AI_PTR(g_modellite_activations_map[0] + 0);
    nl_2_nl_output_array.data = AI_PTR(g_modellite_activations_map[0] + 512);
    nl_2_nl_output_array.data_start = AI_PTR(g_modellite_activations_map[0] + 512);
    gemm_3_output_array.data = AI_PTR(g_modellite_activations_map[0] + 0);
    gemm_3_output_array.data_start = AI_PTR(g_modellite_activations_map[0] + 0);
    nl_3_nl_output_array.data = AI_PTR(g_modellite_activations_map[0] + 256);
    nl_3_nl_output_array.data_start = AI_PTR(g_modellite_activations_map[0] + 256);
    gemm_4_output_array.data = AI_PTR(g_modellite_activations_map[0] + 0);
    gemm_4_output_array.data_start = AI_PTR(g_modellite_activations_map[0] + 0);
    nl_5_output_array.data = AI_PTR(g_modellite_activations_map[0] + 60);
    nl_5_output_array.data_start = AI_PTR(g_modellite_activations_map[0] + 60);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool modellite_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_modellite_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    gemm_1_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_1_weights_array.data = AI_PTR(g_modellite_weights_map[0] + 1024);
    gemm_1_weights_array.data_start = AI_PTR(g_modellite_weights_map[0] + 0);
    gemm_1_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_1_bias_array.data = AI_PTR(g_modellite_weights_map[0] + 233984);
    gemm_1_bias_array.data_start = AI_PTR(g_modellite_weights_map[0] + 233984);
    gemm_2_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_2_weights_array.data = AI_PTR(g_modellite_weights_map[0] + 237056);
    gemm_2_weights_array.data_start = AI_PTR(g_modellite_weights_map[0] + 236032);
    gemm_2_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_2_bias_array.data = AI_PTR(g_modellite_weights_map[0] + 302592);
    gemm_2_bias_array.data_start = AI_PTR(g_modellite_weights_map[0] + 302592);
    gemm_3_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_3_weights_array.data = AI_PTR(g_modellite_weights_map[0] + 303104);
    gemm_3_weights_array.data_start = AI_PTR(g_modellite_weights_map[0] + 303104);
    gemm_3_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_3_bias_array.data = AI_PTR(g_modellite_weights_map[0] + 335872);
    gemm_3_bias_array.data_start = AI_PTR(g_modellite_weights_map[0] + 335872);
    gemm_4_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_4_weights_array.data = AI_PTR(g_modellite_weights_map[0] + 336128);
    gemm_4_weights_array.data_start = AI_PTR(g_modellite_weights_map[0] + 336128);
    gemm_4_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_4_bias_array.data = AI_PTR(g_modellite_weights_map[0] + 339968);
    gemm_4_bias_array.data_start = AI_PTR(g_modellite_weights_map[0] + 339968);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_modellite_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_MODELLITE_MODEL_NAME,
      .model_signature   = AI_MODELLITE_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 309296,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x10f4ec69,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_modellite_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_MODELLITE_MODEL_NAME,
      .model_signature   = AI_MODELLITE_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 309296,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x10f4ec69,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_modellite_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_modellite_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_modellite_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_modellite_create(network, AI_MODELLITE_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_modellite_data_params_get(&params) != true) {
    err = ai_modellite_get_error(*network);
    return err;
  }
#if defined(AI_MODELLITE_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_MODELLITE_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_modellite_init(*network, &params) != true) {
    err = ai_modellite_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_modellite_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_modellite_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_modellite_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_modellite_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= modellite_configure_weights(net_ctx, params);
  ok &= modellite_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_modellite_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_modellite_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_MODELLITE_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

