# Fine-Tuning LLMs for Sentiment Analysis

A comparative study of **TF-IDF + Logistic Regression** and a **fine-tuned DistilBERT** model for binary sentiment classification using IMDb movie reviews.

## Project Overview

This project investigates whether fine-tuning a pre-trained Transformer model can improve sentiment classification compared with a traditional machine learning approach.

### Project Idea

Adapt a pre-trained language model to the sentiment analysis task by fine-tuning **DistilBERT** on IMDb movie reviews, and compare its performance with a **TF-IDF + Logistic Regression** baseline.

## Dataset

- **Dataset:** IMDb Movie Reviews
- **Source:** Hugging Face
- **Total Reviews:** 50,000
- **Training Set:** 25,000
- **Test Set:** 25,000
- **Task:** Binary Sentiment Classification
- **Label 0:** Negative
- **Label 1:** Positive

## Approaches

### 1. Traditional ML Baseline

`IMDb Reviews → TF-IDF → Logistic Regression → Sentiment Prediction`

TF-IDF converts reviews into numerical features, which are then classified using Logistic Regression.

### 2. Fine-Tuned Transformer

`IMDb Reviews → DistilBERT Tokenizer → DistilBERT → Fine-Tuning → Sentiment Prediction`

A pre-trained DistilBERT model is fine-tuned specifically for the IMDb binary sentiment classification task.

## Methodology

1. Load the IMDb dataset.
2. Prepare the reviews and sentiment labels.
3. Build the TF-IDF + Logistic Regression baseline.
4. Tokenize reviews using the DistilBERT tokenizer.
5. Fine-tune the pre-trained DistilBERT model.
6. Evaluate both models using Accuracy, Precision, Recall, and F1-score.
7. Compare the results using performance metrics and confusion matrices.
8. Examine sample predictions through qualitative analysis.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- NumPy
- Pandas
- Google Colab

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | **0.8837** | **0.8844** | **0.8828** | **0.8836** |
| Fine-tuned DistilBERT | **0.9118** | **0.9089** | **0.9154** | **0.9121** |

### Performance Improvement

| Metric | Improvement |
|---|---:|
| Accuracy | **+0.0281** |
| Precision | **+0.0245** |
| Recall | **+0.0326** |
| F1-score | **+0.0285** |

The fine-tuned DistilBERT model performs better than the traditional baseline across all four reported evaluation metrics.

## Confusion Matrix Analysis

The reported confusion matrices show:

| Model | Correct Predictions | Incorrect Predictions |
|---|---:|---:|
| TF-IDF + Logistic Regression | **22,092** | **2,908** |
| Fine-tuned DistilBERT | **22,795** | **2,205** |

DistilBERT produces **703 more correct predictions** and **703 fewer errors** than the baseline in the reported test-set evaluation.

## Qualitative Analysis

Sample IMDb reviews were examined to compare actual sentiment with model predictions. The examples include both positive and negative reviews and demonstrate how the models classify individual review texts.

## Key Findings

- Both approaches successfully perform binary sentiment classification.
- TF-IDF + Logistic Regression provides a strong traditional baseline.
- Fine-tuned DistilBERT achieves better Accuracy, Precision, Recall, and F1-score.
- The confusion-matrix analysis shows fewer classification errors for DistilBERT.
- The results demonstrate the benefit of adapting a pre-trained Transformer to the target sentiment task.

## Conclusion

The project demonstrates that a fine-tuned pre-trained Transformer can provide stronger sentiment classification performance than a traditional TF-IDF-based machine learning approach. **Fine-tuned DistilBERT is the better-performing model in the reported experiment**, showing consistent improvements across all evaluated metrics.

## Project Structure

```text
Fine-Tuning-LLMs-for-Sentiment-Analysis/
│
├── README.md
├── notebooks/
│   └── sentiment_analysis.ipynb
├── results/
├── figures/
└── requirements.txt
