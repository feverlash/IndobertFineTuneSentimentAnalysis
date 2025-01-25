# Indonesian Text Sentiment Analysis 🚀
## 📌 Overview  
This project fine-tunes a **transformer-based model** to analyze sentiment for Indonesian text.  You can download the model here:
[SmSA (IndoNLU) Dataset](https://huggingface.co/feverlash/Indonesian-SentimentAnalysis-Model)

## 📥 Data Collection  
The dataset used for fine-tuning was sourced from **IndoNLU Datasets**, specifically:  
[Indonesian-SentimentAnalysis-Model](https://metatext.io/datasets/smsa-(indonlu))

## 🔄 Data Preparation  
- **Tokenization**:
  - Used **Indobert** for efficient text processing.
- **Train-Test Split**:
  - The Dataset is already splitted into train, validation, and test.

## 🏋️ Fine-Tuning & Results  
The model was fine-tuned using **TensorFlow Hugging Face Transformers**.

### **📊 Evaluation Metrics**
| **Epoch** | **Train Loss** | **Train Accuracy** | **Eval Loss** | **Eval Accuracy** | **Training Time** | **Validation Time** |
|-----------|----------------|---------------------|---------------|-------------------|-------------------|---------------------|
| **1**     | `0.2471`       | `88.15%`           | `0.2107`      | `91.31%`          | `7:55 min`        | `10 sec`            |
| **2**     | `0.1844`       | `90.41%`           | `0.2107`      | `92.39%`          | `7:50 min`        | `10 sec`            |
| **3**     | `0.1502`       | `91.66%`           | `0.2135`      | `93.14%`          | `7:51 min`        | `9 sec`             |
| **4**     | `0.1285`       | `92.50%`           | `0.2192`      | `93.69%`          | `7:50 min`        | `10 sec`            |
| **5**     | `0.1101`       | `93.13%`           | `0.2367`      | `94.14%`          | `7:48 min`        | `9 sec`             |

## ⚙️ Training Parameters  
epochs = 5

learning_rate = 5e-5

seed_val = 42

max_length = 128

batch_size = 32

eval_batch_size = 32
