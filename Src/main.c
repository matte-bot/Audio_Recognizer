/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "string.h"
#include "ai_platform.h"
#include "modellite.h"
#include "model_config.h"
#include "modellite_data.h"
#include "modellite_data_params.h"
#include "stdlib.h"
#include "stdio.h"
#include "cJSON.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */


/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define INPUT_SIZE 2275
#define OUTPUT_SIZE 10
#define EXPECTED_MFCC_SIZE 175

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;
DMA_HandleTypeDef hdma_usart2_rx;
DMA_HandleTypeDef hdma_usart2_tx;

/* USER CODE BEGIN PV */
float input_data[INPUT_SIZE];
float output_data[OUTPUT_SIZE];
float mfcc_data[EXPECTED_MFCC_SIZE];

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */


void run_inference(ai_handle model_handle, ai_buffer *input_buffer, ai_buffer *output_buffer) {
    ai_i32 batch;
    ai_error err;

    // 2. Esegui l'inferenza
    batch = ai_modellite_run(model_handle, input_buffer, output_buffer);

    // 3. Controlla se l'inferenza è riuscita
    if (batch != 1) {
        err = ai_modellite_get_error(model_handle);
        if (err.type != AI_ERROR_NONE){
        char errinf[100];
    	sprintf(errinf,"Errore durante l'inferenza: %d\n", err.type);
    	HAL_UART_Transmit(&huart2,(uint8_t*)errinf, strlen(errinf), HAL_MAX_DELAY);
        Error_Handler();  // Gestione dell'errore: chiama l'Error_Handler() o libera risorse.
    }
    }
    // 4. Post-processa i risultati dell'inferenza
    printf("Risultati dell'inferenza:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
    	char buffer[50];
        sprintf(buffer,"Output[%d]: %f\n", i, ((ai_float *)output_buffer->data)[i]);
        HAL_UART_Transmit(&huart2,(uint8_t*)buffer, strlen(buffer), HAL_MAX_DELAY);
    }
}

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
void get_instrument_name(int index, char *name) {
    switch(index) {
        case 0:
            strcpy(name, "Vocal Acoustic");
            break;
        case 1:
            strcpy(name, "Keyboard Electronic");
            break;
        case 2:
            strcpy(name, "String Acoustic");
            break;
        case 3:
        	strcpy(name, "Keyboard Acoustic");
        	break;
        case 4:
            strcpy(name, "Guitar Electronic");
            break;
        case 5:
            strcpy(name, "Organ Electronic");
            break;
        case 6:
        	strcpy(name, "Guitar Acoustic");
            break;
        case 7:
            strcpy(name, "Flute Acoustic");
            break;
        case 8:
            strcpy(name, "Bass Electronic");
            break;
        case 9:
            strcpy(name, "Brass Acoustic");
            break;

      // Aggiungi ulteriori strumenti secondo necessità
        default:
            strcpy(name, "Sconosciuto");
            break;
    }
}

void load_mfcc_data(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        char errfile[100];
        sprintf(errfile, "Errore nell'apertura del file: %s\n", filename);
        HAL_UART_Transmit(&huart2, (const char*)errfile, strlen(errfile), HAL_MAX_DELAY);
        return;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* data = (char*)malloc(length + 1);
    fread(data, 1, length, file);
    fclose(file);
    data[length] = '\0';

    cJSON* json = cJSON_Parse(data);
    if (json == NULL) {
        HAL_UART_Transmit(&huart2, "Errore nel parsing del JSON\n", 100, HAL_MAX_DELAY);
        free(data);
        return;
    }

    cJSON* mfccs = cJSON_GetObjectItem(json, "mfccs");
    if (mfccs != NULL) {
        cJSON* mfcc = NULL;
        int count = 0; // Conta i MFCC caricati
        cJSON_ArrayForEach(mfcc, mfccs) {
            if (count < EXPECTED_MFCC_SIZE) {
                cJSON* values = cJSON_GetObjectItem(mfcc, "value");
                if (values) {
                    int value_count = cJSON_GetArraySize(values);
                    for (int i = 0; i < value_count && i < EXPECTED_MFCC_SIZE; i++) {
                        float value = (float)cJSON_GetArrayItem(values, i)->valuedouble;
                        mfcc_data[count++] = value; // Riempie mfcc_data
                    }
                }
            }
        }
    }

    cJSON_Delete(json);
    free(data);
}


/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
	HAL_UART_Transmit(&huart2, "Inizio del programma", 100, HAL_MAX_DELAY);

	typedef struct ai_model_config ai_model_config;
	struct ai_model_config {
    	ai_buffer activations;       // Buffer per le attivazioni
    	ai_buffer weights;           // Buffer per i pesi del modello
    	size_t activations_size; 	// Dimensione delle attivazioni
    	size_t weights_size;
    };


    AI_ALIGNED(4) ai_u8 activations[AI_MODELLITE_DATA_ACTIVATIONS_SIZE];
    ai_buffer activations_buffer = AI_BUFFER_INIT(AI_BUFFER_FORMAT_U8,1,1,AI_MODELLITE_DATA_ACTIVATIONS_SIZE,1,activations);

    AI_ALIGNED(4) ai_float input_data[INPUT_SIZE];
    ai_buffer input_buffer = AI_BUFFER_INIT(AI_BUFFER_FORMAT_FLOAT, 1, 1, INPUT_SIZE, 1, input_data);

    AI_ALIGNED(4) ai_float output_data[OUTPUT_SIZE];
    ai_buffer output_buffer = AI_BUFFER_INIT(AI_BUFFER_FORMAT_FLOAT, 1, 1, OUTPUT_SIZE, 1, output_data);

    AI_ALIGNED(4) ai_u8 weights[AI_MODELLITE_DATA_WEIGHTS_SIZE];
    ai_buffer weights_buffer = AI_BUFFER_INIT(AI_BUFFER_FORMAT_U8, 1, 1, AI_MODELLITE_DATA_WEIGHTS_SIZE, 1, weights);

    AI_ALIGNED(4) ai_float mfcc_data[EXPECTED_MFCC_SIZE];
    ai_buffer mfcc_buffer = AI_BUFFER_INIT(AI_BUFFER_FORMAT_FLOAT, 1, 1, OUTPUT_SIZE, 1, mfcc_data);



    /* Create an instance of the model */
    ai_handle model_handle;
    ai_error err;
    ai_model_config model_config;

    model_config.activations = activations_buffer;
    model_config.activations.data = (uint8_t*)malloc(model_config.activations.size);
    model_config.activations.size = 64;

    model_config.weights = weights_buffer;
    model_config.weights.data = (uint8_t*)malloc(model_config.weights.size);
    model_config.weights.size = 256;

    if (model_config.activations.data == NULL || model_config.weights.data == NULL) {
        Error_Handler();
    }


    err = ai_modellite_create(&model_handle, &model_config.activations);
    // Initialize the model
    if (err.type != AI_ERROR_NONE) {
    	char erriniz[100];
    	sprintf(erriniz, "Errore durante l'inizializzazione del modello: %d\n", err.type);
    	HAL_UART_Transmit(&huart2,(uint8_t*)erriniz, strlen(erriniz), HAL_MAX_DELAY);
    }



  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */
  load_mfcc_data("Resources/mfcc.json");

  free(model_config.activations.data); // Libera la memoria delle attivazioni
  free(model_config.weights.data); // Libera la memoria dei pesi

  // Ritorna o termina il programma
  return 0;

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
    while (1)
    {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

        HAL_UART_Receive(&huart2, (uint8_t*)mfcc_data, EXPECTED_MFCC_SIZE * sizeof(float), HAL_MAX_DELAY);

        HAL_UART_Transmit(&huart2, "Dati input ricevuti:", 100, HAL_MAX_DELAY);

        for (int i = 0; i < INPUT_SIZE; i++) {
            // Formatta il valore float in una stringa temporanea
            char buffer[50]; // Dimensione sufficiente per un float
            sprintf(buffer, "%f ", input_data[i]);
            HAL_UART_Transmit(&huart2, (uint8_t*)buffer, strlen(buffer), HAL_MAX_DELAY);
        }

        run_inference(model_handle, &input_buffer, &output_buffer);


        int max_index = 0;
            float max_value = ((ai_float *)output_buffer.data)[0];
            for (int i = 1; i < OUTPUT_SIZE; i++) {
                 if (((ai_float *)output_buffer.data)[i] > max_value) {
                     max_value = ((ai_float *)output_buffer.data)[i];
                     max_index = i;
                    }
                }

        char instrument_name[50];
        get_instrument_name(max_index, instrument_name);


        char message[100];
        sprintf(message, "Strumento riconosciuto: %s\n", instrument_name);
        HAL_UART_Transmit(&huart2, (uint8_t*)message, strlen(message), HAL_MAX_DELAY);

    }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
  HAL_RCC_MCOConfig(RCC_MCO1, RCC_MCO1SOURCE_HSI, RCC_MCODIV_1);
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream5_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream5_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream5_IRQn);
  /* DMA1_Stream6_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream6_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream6_IRQn);

}


/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin : PA8 */
  GPIO_InitStruct.Pin = GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF0_MCO;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

