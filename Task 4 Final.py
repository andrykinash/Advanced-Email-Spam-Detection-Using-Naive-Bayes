import os
import random
import re
from collections import Counter

# Directories 
normal_emails_directory = 'C:/Users/andry/Desktop/492/Assignment 3/normal'
spam_emails_directory = 'C:/Users/andry/Desktop/492/Assignment 3/spam'

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

# Function to preprocess and create bigrams
def process(emails):
    tokenized_emails = []
    for email in emails:
        email = email.lower()  # Convert to lowercase
        email = re.sub(r'\W+', ' ', email)  # Remove special characters
        tokens = email.split()
        bigrams = list(zip(tokens, tokens[1:]))
        tokenized_emails.append(bigrams)
    return tokenized_emails

# Updated bag of words function for bigrams
def bag_of_words(emails, vocabulary):
    vectors = []
    for email in emails:
        email_word_count = Counter(email)
        vector = [email_word_count[bigram] for bigram in vocabulary]
        vectors.append(vector)
    return vectors

# Train a Naive Bayes classifier
def train_naive_bayes(train_vectors, train_labels, smoothing_factor=0.15): #edit smooth factor here
    num_emails = len(train_vectors)
    num_spam = sum(train_labels)
    num_normal = num_emails - num_spam
    
    # Calculate probabilities for each word with smoothing
    word_prob_spam = [(sum(word_vector[i] for i, label in enumerate(train_labels) if label == 1) + smoothing_factor) / 
                      (sum(sum(train_vectors[i]) for i, label in enumerate(train_labels) if label == 1) + (len(vocabulary) * smoothing_factor))
                      for word_vector in zip(*train_vectors)]
    word_prob_normal = [(sum(word_vector[i] for i, label in enumerate(train_labels) if label == 0) + smoothing_factor) / 
                        (sum(sum(train_vectors[i]) for i, label in enumerate(train_labels) if label == 0) + (len(vocabulary) * smoothing_factor))
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
# Preprocessing for bigrams
tokenized_train_normal = process(train_emails_normal)
tokenized_train_spam = process(train_emails_spam)
tokenized_test_normal = process(test_emails_normal)
tokenized_test_spam = process(test_emails_spam)

#updated for bigrams
all_tokenized_emails = tokenized_train_normal + tokenized_train_spam + tokenized_test_normal + tokenized_test_spam
vocabulary = set([bigram for email in all_tokenized_emails for bigram in email])
word_frequency = Counter([bigram for email in all_tokenized_emails for bigram in email])

# Size of the vocabulary and the most common words
vocabulary_size = len(vocabulary)
most_common_words = word_frequency.most_common(10)
vocabulary_size, most_common_words

#--------------------------------BAG OF WORDS INITIALIZING----------------------------------------------

# Convert the training and testing emails into vectors
train_vectors_normal = bag_of_words(tokenized_train_normal, vocabulary)
train_vectors_spam = bag_of_words(tokenized_train_spam, vocabulary)
test_vectors_normal = bag_of_words(tokenized_test_normal, vocabulary)
test_vectors_spam = bag_of_words(tokenized_test_spam, vocabulary)

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