# LLM Project: Sentiment Classification using RoBERTa

# Project Task
The task is to perform sentiment classification on restaurant reviews. The goal is to categorize the sentiment of each review into three categories: positive, neutral, and negative. 

# Dataset
The dataset used is Restaurant_Reviews.csv, which contains the following columns: review and score (1 classified as positive and 0 classified as negative).

# Data Preprocessing:
The dataset is cleaned by removing non-alphabetic characters and stopwords.
Lemmatization is applied to reduce words to their root form.
Text data is vectorized using TF-IDF to capture the term importance.
Sentiment labels are encoded into numerical format (0 for negative, 1 for neutral, 2 for positive).

# Pre-trained Model
For this task, we selected RoBERTa (a variant of BERT) for sequence classification. The model was chosen because:
RoBERTa has shown state-of-the-art performance on various NLP tasks, including text classification, due to its ability to capture deep contextual information.
It can handle long sequences and has been fine-tuned for a variety of applications, making it well-suited for this task.
RoBERTa is a powerful model that works effectively even with moderate-sized datasets, making it ideal for our sentiment classification problem.
We fine-tuned RoBERTa with three sentiment classes (positive, neutral, negative) using the Hugging Face Transformers library.

# Performance Metrics
To evaluate the model's performance, we use the following metrics:
Accuracy: Measures the percentage of correctly predicted labels.
F1-Score: The harmonic mean of precision and recall, focusing on class balance.
Confusion Matrix: Helps to visualize the performance across all classes (positive, neutral, and negative).

Sample Results:
The classification report provides detailed precision, recall, and F1-scores for each sentiment category. The confusion matrix provides a visual representation of the true vs. predicted labels.

# Hyperparameters
The following hyperparameters were crucial during model optimization:

Learning Rate (2e-5): This is a commonly used learning rate for fine-tuning transformer models. A lower learning rate ensures smoother convergence and avoids overshooting the optimal weights.
Batch Size (16): We used a batch size of 16 for both training and evaluation. This was a reasonable size to balance between memory usage and training time.
Epochs (3): The model was trained for 3 epochs. This number strikes a balance between overfitting and underfitting, though adjustments could be made based on validation performance.
Max Sequence Length (512): The maximum length of the input text was truncated or padded to 512 tokens, which is the standard input size for RoBERTa.
Weight Decay (0.01): This is used to regularize the model, helping to prevent overfitting by penalizing large model weights.
Training and Evaluation:
Data Splitting: The dataset was split into 80% training and 20% validation sets.
Tokenization: The text data was tokenized using the RoBERTa tokenizer, which converts the text into token IDs compatible with the model.
Trainer: We used the Hugging Face Trainer class, which simplifies training and evaluation by handling the loop and logging for us.
Evaluation Strategy: The evaluation happens after each epoch, and logs are saved during the training process.

# Relevant Links

https://huggingface.co/puranik/LLM
https://www.kaggle.com/datasets/mawro73/restaurant-reviews-for-nlp?resource=download