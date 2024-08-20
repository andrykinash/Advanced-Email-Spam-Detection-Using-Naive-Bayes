import os
import random
import re
from collections import Counter

# Function to read all text files in a directory
def read_email_files(directory):
    emails = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                emails.append(file.read())
    return emails

# Directory paths
normal_emails_directory = 'C:/Users/andry/Desktop/492/Assignment 3/normal'
spam_emails_directory = 'C:/Users/andry/Desktop/492/Assignment 3/spam'

# Read the content of the normal and spam emails
normal_emails_content = read_email_files(normal_emails_directory)
spam_emails_content = read_email_files(spam_emails_directory)

# Function to preprocess and tokenize emails
def process(emails):
    tokenized_emails = []
    for email in emails:
        email = email.lower()  # Convert to lowercase
        email = re.sub(r'\W+', ' ', email)  # Remove special characters
        email = re.sub(r'\d+', '', email)  # Remove numbers
        tokens = email.split()
        tokenized_emails.append(tokens)
    return tokenized_emails

# 5 normal and 5 spam emails for testing
random.seed(0)
test_indices_normal = random.sample(range(len(normal_emails_content)), 5)
test_indices_spam = random.sample(range(len(spam_emails_content)), 5)

# Extract testing and training datasets
test_emails_normal = [normal_emails_content[i] for i in test_indices_normal]
train_emails_normal = [email for i, email in enumerate(normal_emails_content) if i not in test_indices_normal]
test_emails_spam = [spam_emails_content[i] for i in test_indices_spam]
train_emails_spam = [email for i, email in enumerate(spam_emails_content) if i not in test_indices_spam]

# Preprocess the emails
tokenized_train_normal = process(train_emails_normal)
tokenized_train_spam = process(train_emails_spam)
tokenized_test_normal = process(test_emails_normal)
tokenized_test_spam = process(test_emails_spam)

# Combine all tokenized emails for vocabulary
all_tokenized_emails = tokenized_train_normal + tokenized_train_spam + tokenized_test_normal + tokenized_test_spam

# Create vocabulary and word frequency counter
vocabulary = set([token for email in all_tokenized_emails for token in email])
word_frequency = Counter([token for email in all_tokenized_emails for token in email])

# Size of the vocabulary and the most common words
vocabulary_size = len(vocabulary)
most_common_words = word_frequency.most_common(10)

# Prints to verify everything is working
print("Vocabulary Size:", vocabulary_size)
print("Most Common Words:", most_common_words)
print("\nNumber of Normal Emails in Training:", len(train_emails_normal))
print("Number of Spam Emails in Training:", len(train_emails_spam))
print("Number of Normal Emails in Testing:", len(test_emails_normal))
print("Number of Spam Emails in Testing:", len(test_emails_spam))
