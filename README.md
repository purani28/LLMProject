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
To assess how well the model performs, we used the following metrics:
Accuracy: The percentage of correctly predicted sentiment labels.
F1-Score: A metric that combines precision and recall, particularly useful when dealing with imbalanced classes.
Confusion Matrix: This gives a clear visualization of how well the model is predicting each sentiment class (positive, neutral, and negative).

Sample results from the model's performance:
....
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
Trainer: We used Hugging Face’s Trainer class, which streamlines the training and evaluation process, handling the training loop and logging.
Evaluation Strategy: The model was evaluated after each epoch to monitor progress, and logs were generated during training for easier tracking.

# Relevant Links
https://huggingface.co/puranik/LLM
https://www.kaggle.com/datasets/mawro73/restaurant-reviews-for-nlp?resource=download