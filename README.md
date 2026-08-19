# Fine-tuning LLMs for Sentiment Analysis

## Project Idea
Adapt a pre-trained language model for sentiment classification using the IMDB movie review dataset.

## Dataset
- Dataset: IMDB Movie Reviews
- Training samples: 25,000
- Test samples: 25,000
- Classes: Positive and Negative

## Approach

### 1. Baseline
A traditional machine learning baseline was implemented using:
- TF-IDF Vectorization
- Logistic Regression

### 2. Fine-tuned Model
A pre-trained DistilBERT model was fine-tuned for binary sentiment classification.

## Evaluation Metrics
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 88.37% | 88.44% | 88.28% | 88.36% |
| Fine-tuned DistilBERT | 91.36% | 90.71% | 92.15% | 91.42% |

The fine-tuned DistilBERT model performed better than the TF-IDF + Logistic Regression baseline across the overall evaluation.

## Implementation
The complete implementation is provided in the Google Colab notebook included in this repository.

The notebook covers:
- Dataset loading and inspection
- TF-IDF baseline
- Logistic Regression training
- DistilBERT tokenization
- Fine-tuning
- Model evaluation
- Confusion matrices
- Model comparison
- Sample sentiment predictions
