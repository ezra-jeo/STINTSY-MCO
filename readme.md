# Signals in Silence: Predicting Suicidality from Reddit Posts Using Machine Learning

**STINTSY Machine Project | Term 2, A.Y. 2025-2026**
Group 4 - Parokya ni Ponciano

---

## Project Overview

This project builds and compares machine learning models for detecting suicidal ideation in Reddit posts. Four model families are implemented: K-Nearest Neighbors, Logistic Regression, Multinomial Naive Bayes, and a Neural Network on frozen BERT embeddings, along with a fine-tuned DistilBERT classifier.

---

## File Structure

```
parokya_ni_ponciano_MCO/
├── main.ipynb                                    # Main notebook (full pipeline)
├── feature_engineering.py                        # Feature extraction functions
├── classifiers.py                                # NeuralNetwork class definition
├── utility.py                                    # FeatureTransformer class
├── Suicidal Ideation Detection reddit Dataset.csv  # Raw dataset
└── suicidality_dataset_w_features.csv            # Preprocessed dataset with features
```

---

## Dependencies

### Python Version
Python 3.10 or higher is recommended.

### Core Libraries
```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
emoji
tqdm
```

### Deep Learning
```
torch
transformers
```

### Utilities
```
joblib
textwrap (standard library)
re (standard library)
copy (standard library)
itertools (standard library)
```

### Installation

Install all dependencies at once using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy emoji tqdm torch transformers joblib
```

For GPU support with PyTorch, visit [https://pytorch.org](https://pytorch.org) and install the version matching your CUDA setup.

---

## Running the Notebook

1. Ensure all dependencies are installed.
2. Place all files in the same directory.
3. Open `main.ipynb` and run cells top to bottom.

The notebook is self-contained and runs the full pipeline:
- Data preparation and cleaning
- Feature engineering
- Exploratory data analysis
- Model training and evaluation for all five models
- Final model comparison

> The preprocessed dataset (`suicidality_dataset_w_features.csv`) is included, so the data cleaning and feature engineering sections can be skipped if you only want to run the classification stage.

---

## Saving Model Parameters

By default, model saving is **commented out** to avoid writing files unintentionally. To save a trained model, uncomment the relevant lines in the saving cells described below.

### KNN and FeatureTransformer — `cell after KNN error analysis`
```python
# joblib.dump(transformer, "transformer.joblib")
# joblib.dump(optimized_knn, "knn.joblib")
```
Saves the fitted `FeatureTransformer` (TF-IDF + scaler) and the optimized KNN model. Both are needed together to transform new data and run predictions.

### Logistic Regression — `cell after LogReg error analysis`
```python
# joblib.dump(optimized_logreg, "optimized_logreg.joblib")
```
Saves the optimized Logistic Regression model. The `FeatureTransformer` saved above is also required for inference.

### Multinomial Naive Bayes
Naive Bayes is **not saved** as it trains quickly enough to rerun on its own. No saving cell is provided for this model.

### Neural Network (Frozen BERT + NN) — `cell after NN error analysis`
```python
# torch.save(nn_model.state_dict(), "optimized_neural_net.pth")

# torch.save(train_embeddings, "train_embeddings.pt")
# torch.save(train_embed_labels, "train_labels.pt")
# torch.save(val_embeddings, "val_embeddings.pt")
# torch.save(val_embed_labels, "val_labels.pt")
# torch.save(test_embeddings, "test_embeddings.pt")
# torch.save(test_embed_labels, "test_labels.pt")
```
Saves the trained neural network weights. The embedding files are also provided as optional saves — uncommenting them avoids recomputing BERT embeddings on future runs, which can be time-consuming.

### Fine-tuned DistilBERT — `cell after BERT error analysis`
```python
# torch.save(bert_model.state_dict(), "distilbert_classifier_params.pth")
# model.save_pretrained('./best_finetuned_distilbert')
# tokenizer_ft.save_pretrained('./best_finetuned_distilbert')
```
Saves the fine-tuned DistilBERT model weights. `save_pretrained` saves the full HuggingFace model and tokenizer to a directory, which allows loading with `from_pretrained('./best_finetuned_distilbert')` in the future.

---

## Local Modules

The notebook depends on three local Python files that must be in the same directory:

| File | Purpose |
|---|---|
| `feature_engineering.py` | Functions for extracting linguistic, punctuation, emoji, and emoticon features |
| `classifiers.py` | `NeuralNetwork` class used by the frozen BERT + NN model |
| `utility.py` | `FeatureTransformer` class that combines TF-IDF and engineered features into a single matrix |

---

## Group Members

- Burayag, Ethan Axl
- David Jr., Jose Ponciano
- Del Rosario, Ezra Jeonadab
- Ferrer, Lance Jacob