# Viva Preparation Guide - NLP Mini Project

This file is a ready-to-speak viva guide for your project.  
Use it as:

- a script for likely examiner questions,
- a structure for long answers,
- a defense sheet for follow-up questions.

Project context used here is based on your current docs and outputs:

- `README.md`
- `LOAD_REPORT_AND_EPOCH_LOG_EXPLANATION.md`
- notebook outputs (accuracy/F1, confusion matrix, ROC AUC, training loss, word cloud)

---

## 1) 60-second project opening (say this first)

**Question:** "Explain your project in one minute."

**Answer (high-scoring):**

"My project is binary sentiment classification on IMDB movie reviews using Transformer models. I fine-tuned two pretrained models, BERT and DistilBERT, on a shuffled subset of IMDB (1000 train, 400 test). I used tokenization with max length 128, trained with AdamW, and evaluated using accuracy, F1, confusion matrix, ROC-AUC, and classification report. I also built explainability using attention-based important tokens and deployed two Gradio apps: one for single-model prediction and one for side-by-side comparison. In my run, BERT got 0.7725 accuracy and DistilBERT got 0.79, with ROC AUC around 0.92 for BERT. The key learning is practical trade-off: BERT gives strong representation, while DistilBERT can be faster and still competitive."

---

## 2) Problem and Motivation

### Q1. Why did you choose sentiment analysis?

**Answer:**  
Sentiment analysis is a real business problem in review analytics, recommendation monitoring, and customer feedback mining. Movie reviews are unstructured and context-heavy, so this is a good NLP benchmark to demonstrate model understanding beyond keywords.

### Q2. Why IMDB dataset?

**Answer:**  
IMDB is a standard binary sentiment benchmark with clear labels and large text samples. It is widely used for fair comparison of NLP models and suitable for mini-project demonstration.

### Q3. What is the exact problem formulation?

**Answer:**  
Given a review text, predict one of two classes: `positive` or `negative`. This is supervised binary text classification.

---

## 3) Data and Preprocessing

### Q4. Why did you use only 1000 train and 400 test?

**Answer:**  
I used a smaller subset to reduce training time for classroom/demo constraints. I shuffled with seed 42 for reproducibility. I clearly mention that full-dataset training is a future improvement for stronger stability.

### Q5. Why max length = 128?

**Answer:**  
It balances context retention and compute cost. Larger sequence length gives more context but increases memory/time quadratically in attention-heavy models. For a mini-project, 128 is a practical compromise.

### Q6. What does `attention_mask` do?

**Answer:**  
It tells the model which tokens are real and which are padding. This prevents padded tokens from affecting attention scores and final prediction.

### Q7. Why rename `label` to `labels`?

**Answer:**  
Hugging Face sequence-classification models expect the target key as `labels` during training so the model can compute loss automatically.

---

## 4) Model Choice and Architecture

### Q8. Why compare BERT and DistilBERT?

**Answer:**  
To show quality-speed trade-off. BERT is larger and often stronger context-wise, while DistilBERT is smaller/faster. Real deployments often require this trade-off analysis, not just highest single metric.

### Q9. Difference between BERT and DistilBERT in one line?

**Answer:**  
DistilBERT is a compressed/distilled version of BERT with fewer parameters and faster inference, usually with small accuracy trade-off.

### Q10. Why use pretrained models instead of training from scratch?

**Answer:**  
Pretraining provides language knowledge learned from massive corpora. Fine-tuning needs far less data and compute than training from scratch, and gives better results for practical timelines.

---

## 5) Training and Optimization

### Q11. Explain your training loop briefly.

**Answer:**  
For each batch: tokenize data already prepared -> move tensors to device -> forward pass -> get loss -> `backward()` -> `optimizer.step()` -> `optimizer.zero_grad()`. I store average loss per epoch for learning-curve analysis.

### Q12. Why AdamW with learning rate 2e-5?

**Answer:**  
AdamW is standard for Transformer fine-tuning because of stable convergence with decoupled weight decay. 2e-5 is a commonly reliable starting LR for pretrained BERT-family models.

### Q13. Why call `model.train()` and `model.eval()`?

**Answer:**  
`train()` enables training behavior (like dropout). `eval()` disables training-only randomness and ensures stable inference. This separation is required for correct training and evaluation.

---

## 6) Load Report Questions (very likely in viva)

### Q14. What does `UNEXPECTED` mean in load report?

**Answer:**  
Those weights exist in the checkpoint but are not used by the currently loaded architecture. In transfer learning from base checkpoints to a task-specific head, this is usually expected and safe.

### Q15. What does `MISSING` mean?

**Answer:**  
These parameters are required by the current model class but not found in the checkpoint. So they are newly initialized and must be learned during fine-tuning.

### Q16. Are `UNEXPECTED` and `MISSING` errors?

**Answer:**  
Not automatically. They are normal when intentionally loading a base pretrained model into a different downstream task architecture (like sequence classification). They are only problematic if identical architecture was expected.

### Q17. Explain BERT and DistilBERT load reports in your project.

**Answer:**  
For BERT, pretraining-head keys were unexpected and `classifier.*` was missing. For DistilBERT, vocab/projection-related keys were unexpected and `pre_classifier.*` + `classifier.*` were missing. In both cases, encoder weights loaded, task head initialized new, then fine-tuned on IMDB.

---

## 7) Metrics and Graph Interpretation

### Q18. Why use both Accuracy and F1?

**Answer:**  
Accuracy gives overall correctness, but F1 balances precision and recall. F1 is better when class-wise error balance matters. Using both avoids one-sided interpretation.

### Q19. Interpret your model comparison bars.

**Answer:**  
In this run DistilBERT is slightly higher than BERT in both accuracy and F1, showing that compact models can be highly competitive under constrained training conditions.

### Q20. Explain your confusion matrix numbers.

**Answer:**  
For BERT on 400 samples: TN=119, FP=77, FN=14, TP=190. Accuracy is (119+190)/400 = 0.7725. The model catches positives well (high TP, low FN) but makes more false positives on actual negatives.

### Q21. Why is ROC-AUC important if you already have accuracy?

**Answer:**  
Accuracy is threshold-dependent (single cutoff), while ROC-AUC evaluates ranking quality across all thresholds. AUC ~0.92 indicates strong class separability even if fixed-threshold accuracy is moderate.

### Q22. Why does loss and accuracy sometimes disagree?

**Answer:**  
Loss measures confidence-weighted training error; accuracy is discrete correctness on test data. A model can have slightly higher training loss but better generalization on held-out test samples.

### Q23. Explain the single-point loss plots.

**Answer:**  
Only one epoch was run, so each model has one loss point (BERT ~0.595, DistilBERT ~0.610). It is a snapshot, not a trend. More epochs are needed to claim convergence behavior.

### Q24. What is the purpose of the word cloud?

**Answer:**  
EDA only. It shows frequent words in sampled reviews and validates domain vocabulary (movie, film, story). It is not a predictive metric.

### Q25. Why is `br` large in word cloud?

**Answer:**  
IMDB text often contains HTML line-break artifacts. If not cleaned, they appear as `br`. This suggests a preprocessing improvement: strip HTML tags before exploratory visualization and possibly before modeling experiments.

---

## 8) Explainability and UI

### Q26. How did you do explainability?

**Answer:**  
I extracted last-layer attentions, averaged heads, computed token-level scores, and displayed top influential tokens. This gives interpretable hints of what text parts influenced prediction.

### Q27. Is attention a perfect explanation method?

**Answer:**  
No. Attention gives useful insight but is not a guaranteed causal explanation. I present it as practical interpretability, not absolute proof.

### Q28. Why Gradio in an NLP project?

**Answer:**  
It converts model code into an interactive application, enabling real-time demonstration, qualitative testing, and communication of results to non-technical users.

---

## 9) Limitations and Improvements (examiners love this)

### Q29. What are your key limitations?

**Answer:**  
Small training subset, only one epoch, no dedicated validation split, limited hyperparameter search, and attention-based explanation limits.

### Q30. If given more time, what will you improve first?

**Answer:**  
Increase data size and epochs, add validation and early stopping, perform LR/batch-size tuning, clean HTML artifacts, and evaluate stability across multiple random seeds.

### Q31. How will you improve reliability of conclusions?

**Answer:**  
Run repeated experiments with different seeds, report mean and standard deviation, and compare both models on same train/val/test protocol.

---

## 10) Tough Follow-Up Questions and Full-Marks Responses

### Q32. "Why did DistilBERT beat BERT in your run?"

**Answer:**  
Because the setup is small-data + one-epoch. In such constrained settings, optimization dynamics and regularization can let DistilBERT generalize slightly better on the sampled test split. This does not prove DistilBERT is universally better; it proves model ranking is experiment-dependent and should be validated across larger runs.

### Q33. "Can you deploy this in production directly?"

**Answer:**  
Not directly. It is a strong prototype. For production I would add robust preprocessing, drift monitoring, bias checks, latency benchmarks, model versioning, and secure API deployment with logging and rollback.

### Q34. "How do you justify your methodology academically?"

**Answer:**  
I followed a complete supervised NLP pipeline: benchmark dataset, pretrained transfer learning, controlled preprocessing, comparative model study, multi-metric evaluation, error-pattern analysis via confusion matrix, threshold-agnostic ROC-AUC, interpretability component, and practical UI demonstration.

---

## 11) Scoring Tips for Viva Delivery

- Start each answer with a **definition**, then project-specific **evidence** (your numbers), then **conclusion**.
- Quote your key numbers confidently:
  - BERT accuracy = **0.7725**
  - DistilBERT accuracy = **0.79**
  - BERT AUC ≈ **0.92**
  - Epoch-1 loss: BERT ~**0.595**, DistilBERT ~**0.610**
- Never say "perfect model"; say "strong result under current constraints."
- When asked about warnings (`UNEXPECTED`/`MISSING`), say clearly: "**expected in transfer learning**."
- Mention both **strengths and limitations**: this gives maturity and usually improves marks.

---

## 12) 30-second closing statement

"This project demonstrates an end-to-end NLP system, not only model training. I compared BERT and DistilBERT fairly, interpreted outputs with multiple metrics and graphs, explained load-report behavior correctly, added token-level interpretability, and deployed with Gradio for usability. My current results show DistilBERT slightly ahead on this constrained setup, while also highlighting that broader training and validation are needed for final model selection."

