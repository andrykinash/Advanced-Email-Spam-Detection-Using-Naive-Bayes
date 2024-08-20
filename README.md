# Advanced Email Spam Detection Using Naive Bayes

## Overview
This project implements a robust and comprehensive spam email detection system using Naive Bayes classification. It includes a multi-stage approach to progressively refine the model, ensuring high accuracy and reliability in identifying spam emails. The project leverages Natural Language Processing (NLP) techniques, such as tokenization, bag of words, and bigrams, to preprocess email data before feeding it into the classifier.

## Project Structure
The project is divided into four main tasks, each building on the previous one to improve the performance of the spam classifier:

1. **Task 1: Data Preprocessing and Initial Model**
   - **Objective**: Read and preprocess email data from normal and spam directories.
   - **Process**: Tokenization of emails, removal of unwanted content, and splitting the data into training and testing sets.
   - **Output**: Vocabulary size and the most common words used, alongside initial model performance.

2. **Task 2: Word to Vector Conversion and Naive Bayes Training**
   - **Objective**: Convert emails into a vector space model and train a Naive Bayes classifier.
   - **Process**: Implemented word-to-vector conversion and calculated word probabilities for training the model.
   - **Output**: Precision, recall, F1 score, and identification of false positives/negatives.

3. **Task 3: Bag of Words Approach**
   - **Objective**: Enhance the model using a bag of words technique.
   - **Process**: Shifted to a bag of words model to improve the accuracy of the classifier.
   - **Output**: Reduced false positives and improved overall accuracy of the model.

4. **Task 4: Bigrams and Smoothing**
   - **Objective**: Further refine the model using bigrams and custom smoothing techniques.
   - **Process**: Introduced bigrams and adjusted the smoothing factor to reduce overfitting.
   - **Output**: Finalized the model with balanced precision and recall, eliminating the elusive false negatives.

## Key Features
- **Multi-Stage Implementation**: Each task progressively refines the model, offering insights into the effectiveness of different NLP techniques.
- **Naive Bayes Classifier**: Utilizes the Naive Bayes algorithm, a proven method for spam detection, with custom optimizations.
- **Comprehensive Evaluation**: The project includes detailed performance metrics, such as precision, recall, F1 score, and error analysis.
- **No External Dependencies**: The project relies solely on the normal and spam email directories provided, making it easy to set up and use.

## Getting Started
### Prerequisites
- Python 3.x
- Basic understanding of Python libraries such as `os`, `random`, `re`, `collections`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
