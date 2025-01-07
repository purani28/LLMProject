# LLM Project: Sentiment Classification using RoBERTa

# Project Task
The task is to perform sentiment classification on restaurant reviews. The goal is to categorize the sentiment of each review into three categories: positive, neutral, and negative. 

# Dataset
The dataset used is Restaurant_Reviews.csv, which contains two main columns:
Review: The actual review text written by the customer.
Score: A numerical score given by the customer.

We preprocess the data by:
Removing non-alphabetic characters.
Eliminating stopwords (common words like "the", "is", etc. that don't contribute much to sentiment).
Lemmatizing the words to reduce them to their root forms (for example, "running" becomes "run").

Next, we vectorize the text using TF-IDF to capture the importance of each term in the review. Finally, we convert the sentiment labels into numerical values:
0 for negative sentiment
1 for neutral sentiment
2 for positive sentiment.

# Pre-trained Model
For this sentiment classification task, we selected RoBERTa (a variant of BERT). We chose RoBERTa because:
It has demonstrated state-of-the-art performance on a wide range of NLP tasks, including text classification.
It excels at understanding the context of long text sequences, which is important for analyzing restaurant reviews.
RoBERTa works effectively even with moderate-sized datasets, making it an ideal choice for this project.
We fine-tuned the pre-trained RoBERTa model on our dataset, adapting it to classify reviews into three sentiment categories using the Hugging Face Transformers library.

# Performance Metrics
For label 0 (negative):
Precision: 0.90 – 90% of the time when the model predicted NEGATIVE, it was correct.
Recall: 0.90 – 90% of the actual negative sentiments were correctly identified.
F1-score: 0.90 – This is the combined metric of precision and recall.

For label 1 (positive):
Similar metrics of 0.90, indicating that the model performs equally well for both positive and negative sentiments.

Confusion Matrix:

The confusion matrix shows how well the model is performing:
True Negatives (452): Correctly predicted negative sentiments.
False Positives (48): Predicted positive when the sentiment was actually negative.
False Negatives (50): Predicted negative when the sentiment was actually positive.
True Positives (450): Correctly predicted positive sentiments.

Overall, the model is performing well:
The accuracy is 90%, meaning that 90% of the time, the model predicts the correct sentiment.
The F1-score and other metrics are balanced between positive and negative classes, indicating the model doesn't favor one sentiment over the other.

# Hyperparameters

The following hyperparameters played a key role in optimizing the model:
Learning Rate (2e-5): This is a standard learning rate for fine-tuning transformer models. A smaller learning rate ensures gradual convergence and helps avoid overshooting the optimal solution.
Batch Size (16): We used a batch size of 16, which is a good balance between memory usage and model performance.
Epochs (3): The model was trained for 3 epochs. This number provides a balance between preventing overfitting and underfitting, but could be adjusted based on validation performance.
Max Sequence Length (512): We set the maximum input length to 512 tokens, which is the standard for RoBERTa models. Longer reviews were truncated, and shorter ones were padded.
Weight Decay (0.01): Weight decay was used to regularize the model, preventing overfitting by penalizing large weights.

Training and Evaluation Process:
Data Splitting: The dataset was split into an 80% training set and a 20% validation set.
Tokenization: The reviews were tokenized using RoBERTa’s tokenizer, which converts the text into tokens that the model can understand.
Trainer: Used Hugging Face’s Trainer class, which streamlines the training and evaluation process, handling the training loop and logging.
Evaluation Strategy: The model was evaluated after each epoch to monitor progress, and logs were generated during training for easier tracking.

Results: 
For label 0 (negative):
Precision: 0.84 – 84% of the time when the model predicted NEGATIVE, it was correct.
Recall: 0.88 – 88% of the actual negative instances were correctly identified.
F1-Score: 0.86 – This is the combined metric of precision and recall for label 0, indicating a strong performance.

For label 1 (positive):
Precision: 0.87 – 87% of the time when the model predicted POSITIVE, it was correct.
Recall: 0.82 – 82% of the actual positive instances were correctly identified.
F1-Score: 0.85 – This is the combined metric of precision and recall for label 1, showing that the model performs well for the positive class as well.

Confusion Matrix:
The confusion matrix gives us a detailed breakdown of the model's performance:
True Negatives (91): Correctly predicted negative instances (label 0).
False Positives (12): Predicted positive when the actual label was negative.
False Negatives (17): Predicted negative when the actual label was positive.
True Positives (80): Correctly predicted positive instances (label 1).

Overall Model Performance:
Accuracy: 0.85 – 85% of the time, the model correctly predicted the sentiment.
Macro Average:
Precision: 0.86
Recall: 0.85
F1-Score: 0.85
The metrics for both classes (negative and positive) are close, showing that the model is balanced and does not favor one class over the other.


# Relevant Links
https://huggingface.co/puranik/LLM
https://www.kaggle.com/datasets/mawro73/restaurant-reviews-for-nlp?resource=download
Model Results - https://drive.google.com/file/d/1yofFDytjDqGvvfDX3BQD3xJ3X9ZK6SDI/view?usp=sharing