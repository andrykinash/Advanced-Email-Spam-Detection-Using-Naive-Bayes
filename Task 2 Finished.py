import os
import random
import re
from collections import Counter

# Directories 
normal_emails_directory = 'C:/Users/andry/Desktop/Naive-Bayes/normal'
spam_emails_directory = 'C:/Users/andry/Desktop/Naive-Bayes/spam'

# ------------------------------------------ALL MY FUNCTIONS-----------------------------------------------

# Read directories
def read_email_files(directory):
    emails = []
    file_names = []
    for filename in sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0])):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                emails.append(file.read())
                file_names.append(filename)
    return emails, file_names

# Function to preprocess and tokenize emails
def process(emails):
    tokenized_emails = []
    for email in emails:
        email = email.lower()  # Convert to lowercase
        email = re.sub(r"[\"']", '', email)  # Remove quotation marks specifically
 #       email = re.sub(r'\W+', ' ', email)  # Remove all special characters
 #       email = re.sub(r'\d+', '', email)  # Remove numbers
        tokens = email.split()
        tokenized_emails.append(tokens)
    return tokenized_emails

# Encode the emails into a vector space model
def word_to_vector(emails, vocabulary):
    vectors = []
    vocab_list = list(vocabulary)
    for email in emails:
        vector = [1 if word in email else 0 for word in vocab_list]
        vectors.append(vector)
    return vectors

# Train a Naive Bayes classifier
def train_naive_bayes(train_vectors, train_labels):
    num_emails = len(train_vectors)
    num_spam = sum(train_labels)
    num_normal = num_emails - num_spam
    
    # Calculate probabilities for each word
    word_prob_spam = [(sum(word_vector[i] for i, label in enumerate(train_labels) if label == 1) + 1) / (num_spam + 2)
                      for word_vector in zip(*train_vectors)]
    word_prob_normal = [(sum(word_vector[i] for i, label in enumerate(train_labels) if label == 0) + 1) / (num_normal + 2)
                        for word_vector in zip(*train_vectors)]
    
    return word_prob_spam, word_prob_normal

# Predict whether an email is spam or normal using the Naive Bayes classifier
def predict_naive_bayes(vector, word_prob_spam, word_prob_normal):
    spam_score = sum(w * v for w, v in zip(word_prob_spam, vector))
    normal_score = sum(w * v for w, v in zip(word_prob_normal, vector))
    return 1 if spam_score > normal_score else 0

#---------------------------------------------PROCESSING----------------------------------------------------

# Read the content of the normal and spam emails and get file names
normal_emails_content, normal_file_names = read_email_files(normal_emails_directory)
spam_emails_content, spam_file_names = read_email_files(spam_emails_directory)

# 5 normal and 5 spam emails for testing
#random.seed(0) #this ensures we get the same results for testing when commented out every run of the program gives random unique training and testing files
test_indices_normal = random.sample(range(len(normal_emails_content)), 5)
test_indices_spam = random.sample(range(len(spam_emails_content)), 5)

# Extract testing and training datasets
test_emails_normal = [normal_emails_content[i] for i in test_indices_normal]
train_emails_normal = [email for i, email in enumerate(normal_emails_content) if i not in test_indices_normal]
test_emails_spam = [spam_emails_content[i] for i in test_indices_spam]
train_emails_spam = [email for i, email in enumerate(spam_emails_content) if i not in test_indices_spam]
# Preprocessing
tokenized_train_normal = process(train_emails_normal)
tokenized_train_spam = process(train_emails_spam)
tokenized_test_normal = process(test_emails_normal)
tokenized_test_spam = process(test_emails_spam)

all_tokenized_emails = tokenized_train_normal + tokenized_train_spam + tokenized_test_normal + tokenized_test_spam
vocabulary = set([token for email in all_tokenized_emails for token in email])
word_frequency = Counter([token for email in all_tokenized_emails for token in email])

# Size of the vocabulary and the most common words
vocabulary_size = len(vocabulary)
most_common_words = word_frequency.most_common(10)
vocabulary_size, most_common_words

#--------------------------------WORD TO VECTOR INITIALIZING----------------------------------------------

# Convert the training and testing emails into vectors
train_vectors_normal = word_to_vector(tokenized_train_normal, vocabulary)
train_vectors_spam = word_to_vector(tokenized_train_spam, vocabulary)
test_vectors_normal = word_to_vector(tokenized_test_normal, vocabulary)
test_vectors_spam = word_to_vector(tokenized_test_spam, vocabulary)

# Combine the training vectors and labels
train_vectors = train_vectors_normal + train_vectors_spam
train_labels = [0] * len(train_vectors_normal) + [1] * len(train_vectors_spam)

# Train the classifier
word_prob_spam, word_prob_normal = train_naive_bayes(train_vectors, train_labels)

# Combine the testing vectors and labels, and make predictions
test_vectors = test_vectors_normal + test_vectors_spam
test_labels = [0] * len(test_vectors_normal) + [1] * len(test_vectors_spam)
predictions = [predict_naive_bayes(vector, word_prob_spam, word_prob_normal) for vector in test_vectors]

#------------------------------------METRICS AND PRINTS----------------------------------------------------

false_positives = []
false_negatives = []

# Calculate performance metrics
tp = sum(1 for i in range(len(test_labels)) if test_labels[i] == predictions[i] == 1)
tn = sum(1 for i in range(len(test_labels)) if test_labels[i] == predictions[i] == 0)
fp = sum(1 for i in range(len(test_labels)) if test_labels[i] == 0 and predictions[i] == 1)
fn = sum(1 for i in range(len(test_labels)) if test_labels[i] == 1 and predictions[i] == 0)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Combine test file names
test_file_names = ['normal_' + normal_file_names[i] for i in test_indices_normal] + \
                  ['spam_' + spam_file_names[i] for i in test_indices_spam]

# Identify false positives and false negatives
for i in range(len(test_labels)):
    if test_labels[i] == 0 and predictions[i] == 1:
        false_positives.append(test_file_names[i])
    elif test_labels[i] == 1 and predictions[i] == 0:
        false_negatives.append(test_file_names[i])

# Print the performance metrics
print("\nTrue Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Print the file names of the false positives and false negatives
print("\nFalse Positive Files:", false_positives)
print("False Negative Files:", false_negatives)
