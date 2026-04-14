# Project Report  
## Fine-Tuned Transformer Models for Sentiment Analysis with Explainable Attention (Colab)

**Course / Institution:** *(fill in)*  
**Author:** *(fill in)*  
**Date:** April 2026  

---

## Abstract

This project implements a **binary sentiment analysis** system for movie-review text using **fine-tuned transformer models**. Two pretrained architectures—**BERT** (`bert-base-uncased`) and **DistilBERT** (`distilbert-base-uncased`)—are adapted on a subset of the **IMDB** dataset using a **manual PyTorch training loop** in **Google Colab**. Performance is measured with **accuracy** and **F1 score**; behavior is visualized with a **confusion matrix**. An **attention-based** heuristic highlights subword tokens associated with the model’s processing of each input. A **Gradio** interface allows interactive testing and model comparison. The work demonstrates an end-to-end **natural language processing (NLP)** pipeline combining **deep learning**, **evaluation**, and **lightweight explainability**, suitable for educational and prototyping contexts.

---

## 1. Introduction

Sentiment analysis is a core NLP task: given a piece of text, the system predicts whether the expressed opinion is **positive** or **negative**. Classical approaches relied on bag-of-words features or shallow models; **transformer** architectures now dominate because they capture **context** (e.g., “not good” vs. “good”) through **self-attention** and deep contextualized representations.

**Bidirectional Encoder Representations from Transformers (BERT)** and its distilled variant **DistilBERT** provide strong off-the-shelf encoders. For a specific domain or label set, **fine-tuning**—continuing training on task-specific labeled data—typically yields much better results than using the raw pretrained head.

This report documents a **compact Colab implementation** (`colab_sentiment_fast_full.py`) that fine-tunes both models on IMDB reviews, evaluates them, visualizes errors via a confusion matrix, exposes **token-level attention summaries** for transparency, and ships an interactive **Gradio** demo with a shareable link.

---

## 2. Overview

The pipeline proceeds in stages:

1. **Install** Python dependencies (`transformers`, `datasets`, `scikit-learn`, `gradio`, `matplotlib`, `seaborn`, `torch`).
2. **Load** the IMDB dataset and select smaller train/test subsets for **fast** experimentation.
3. **Tokenize** text with model-specific tokenizers; align labels with Hugging Face conventions (`labels` column).
4. **Train** each model with **AdamW**, **cross-entropy** loss (provided by the library when `labels` are passed), and **one epoch** per model by default.
5. **Evaluate** on the held-out subset; print accuracy and F1.
6. **Plot** a confusion matrix for BERT on a sample of raw test reviews.
7. **Explain** predictions using **last-layer attention** (averaged over heads and queries) to rank subword tokens.
8. **Deploy** a small web UI via **Gradio** (`share=True` for Colab).

The implementation intentionally avoids the Hugging Face **`Trainer`** API in the Colab script to keep the **optimization loop** explicit for learning and debugging; a separate local script may use `Trainer` for convenience.

---

## 3. Motivation

- **Educational clarity:** A visible training loop helps connect **loss**, **gradients**, and **metrics**.
- **Practical NLP:** Sentiment analysis appears in **reviews**, **social media**, and **customer feedback** automation.
- **Trust and debugging:** Showing **confidence** and **token-level hints** supports manual inspection, even if attention is not a full causal explanation.
- **Resource constraints:** **Smaller subsets** and **shorter max sequence length** make repeated runs feasible on free Colab GPUs within reasonable time.

---

## 4. Objectives

1. Fine-tune **BERT** and **DistilBERT** for **binary sentiment** on IMDB data.
2. Quantify performance using **accuracy** and **F1 score**.
3. Visualize classification behavior with a **confusion matrix**.
4. Provide a **simple explainability** view based on **attention weights**.
5. Deliver an **interactive** **Gradio** application for qualitative testing and model comparison.

---

## 5. Scope

### In scope

- Binary classification (**negative** vs. **positive**) on **English** movie reviews.
- Two models: **bert-base-uncased**, **distilbert-base-uncased**.
- Subsampled IMDB train/test sets and **max sequence length 128** (fast Colab preset).
- Attention-based **token ranking** (heuristic, last layer).
- Interactive UI on **Colab** via Gradio **share link**.

### Out of scope

- Full training on the complete IMDB corpus or **large-scale** hyperparameter search.
- **Multilingual** or **multi-class** sentiment.
- Production deployment (scaling, authentication, monitoring).
- Rigorous **model-agnostic** explainability (e.g., SHAP, integrated gradients) as primary deliverables.

---

## 6. System Specifications

### 6.1 Software

| Component | Role |
|-----------|------|
| Python 3.x | Runtime |
| PyTorch | Tensors, autograd, GPU execution |
| Hugging Face Transformers | `AutoTokenizer`, `AutoModelForSequenceClassification` |
| Hugging Face Datasets | `load_dataset("imdb")` |
| scikit-learn | Accuracy, F1, confusion matrix |
| Matplotlib / Seaborn | Confusion matrix plot |
| Gradio | Web UI |

### 6.2 Hardware (typical Colab)

- **GPU** runtime (e.g., T4): recommended for acceptable training time.
- **CPU** only: runs but training is significantly slower.

### 6.3 Models

| Model | Identifier | Notes |
|-------|------------|--------|
| BERT | `bert-base-uncased` | 12 layers, 110M parameters (base); strong baseline. |
| DistilBERT | `distilbert-base-uncased` | Fewer layers; faster, often slightly lower accuracy than BERT. |

Both use a **classification head** with **two** output logits (negative / positive).

### 6.4 Dataset and preprocessing

- **IMDB**: supervised binary labels with review text.
- **Train / test subset:** 1000 / 400 examples (fast configuration); shuffle with **seed 42**.
- **Tokenization:** padding to **max_length 128**, truncation enabled.
- **Labels:** column renamed from `label` to **`labels`** for compatibility with the Transformers loss.

### 6.5 Training hyperparameters (default in script)

| Parameter | Value |
|-----------|--------|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 1 per model |
| Loss | Cross-entropy (from model when `labels` supplied) |

### 6.6 Explainability

- **`attn_implementation="eager"`** so attention tensors are materialized (avoids issues with some optimized attention backends that omit attention weights).
- Last layer attention averaged over heads and query positions; top **10** subword tokens reported.

### 6.7 User interface

- **Inputs:** free text; model choice (**BERT** or **DistilBERT**).
- **Outputs:** predicted sentiment and **confidence**; list of token scores.
- **Error handling:** exceptions return message + traceback to aid debugging on Colab.

---

## 7. Results

*(Paste your actual Colab output below after each run.)*

### 7.1 Quantitative metrics

After training, the script prints dictionaries similar to:

```text
BERT: {'accuracy': <value>, 'f1': <value>}
DistilBERT: {'accuracy': <value>, 'f1': <value>}
```

**Table 1 — Example placeholder (replace with your numbers)**

| Model | Accuracy | F1 Score |
|--------|----------|----------|
| BERT | *(fill in)* | *(fill in)* |
| DistilBERT | *(fill in)* | *(fill in)* |

**Interpretation (generic):** DistilBERT is often **slightly** below BERT on accuracy/F1 but **faster** at inference. Exact ordering depends on subset size, random seed, and number of epochs.

### 7.2 Confusion matrix

The script displays a **heatmap** for BERT on **150** test examples (configurable). Inspect **false positives** and **false negatives** to see whether the model confuses negative text with positive or vice versa.

**Figure 1:** *(Insert screenshot of the confusion matrix from Colab.)*

### 7.3 Qualitative UI behavior

- Short clearly **positive** or **negative** sentences tend to yield **higher confidence**.
- **Ambiguous** or **negated** sentences may show **lower confidence** or misclassification—useful cases for discussion in conclusions.

**Example (illustrative only):**

| Input (example) | Typical observation |
|-----------------|---------------------|
| “I loved this film.” | Often predicted **Positive** with higher confidence. |
| “A complete waste of time.” | Often **Negative**. |
| “It was not a good movie.” | May be **harder**; negation can confuse shallow patterns. |

### 7.4 Explainability output

The UI lists subword tokens (WordPiece) and numeric scores. **Higher** scores indicate positions receiving more aggregated attention mass under this heuristic—not guaranteed “importance” in a causal sense.

---

## 8. Conclusions

This project successfully demonstrates **fine-tuning** transformer encoders for **sentiment analysis** using a **transparent training loop**, **standard metrics**, and a **confusion matrix** for error analysis. **DistilBERT** offers a **lighter** alternative to **BERT** at the cost of some accuracy, which is a useful trade-off for latency-sensitive applications. The **attention-based** token ranking provides **intuitive** feedback for debugging and learning, while acknowledging limitations versus dedicated interpretability methods.

Future work could include **longer training**, **more data**, **hyperparameter tuning**, **longer max sequence length**, saving checkpoints for reuse, and integrating **stronger explainability** tools. Deploying behind an API with authentication would be required for production use.

---

## 9. References

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** *NAACL-HLT.*  
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.** *NeurIPS EMC² Workshop.*  
3. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). **Learning Word Vectors for Sentiment Analysis.** *ACL.* (IMDB dataset.)  
4. Hugging Face. **Transformers** documentation: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)  
5. Hugging Face. **Datasets** documentation: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)  
6. Paszke, A., et al. (2019). **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** *NeurIPS.*  
7. Gradio. **Building machine learning web apps:** [https://www.gradio.app/](https://www.gradio.app/)  

---

*End of report.*
