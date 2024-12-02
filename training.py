import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

#import data
data = pd.read_csv('Combined Data.csv')
print(data.head())

#unncoment this to see distribution of status
# sns.countplot(data['status'])
# plt.xticks(rotation=45)
# plt.title('Distribution of Mental Health Statuses')
# plt.show()
# X = data['statement']
# y = data['status']

# get rid of all columns with missing values
data = data.dropna(subset=['statement'])

# converts statement to numbers
tfidf = TfidfVectorizer(
    lowercase=True,                
    stop_words='english',          # gets rids of stopwords(dont contribute to meaning ie conjuctions and stuff)
    max_features=5000,             # too many unique words, do this to improve model performance    
    ngram_range=(1, 2)             # allow model to have 1 or 2 words in particular, ex: "happy" and "not happy"
    
) 


#TF-IDF is a statistical measure that evaluates how important a word is in a document relative to a collection of documents. 
# This is especially useful in machine learning models, as it gives more weight to important or frequently occurring words in a statement but reduces the weight of common words like "the," "and," etc.


# Term Frequency (TF): How frequently a word appears in a statement.
# Inverse Document Frequency (IDF): A measure of how rare a word is across all statements in the dataset. Rare words are given more weight.
# Apply the vectorizer to the 'statement' column to get the TF-IDF features
X = tfidf.fit_transform(data['statement'])

# 2. Encode the 'status' column (target labels) using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['status'])




#create a new column for the length of each statement
data['statement_length'] = data['statement'].apply(lambda x: len(x.split()))

#plot the distribution of statement lengths
plt.figure(figsize=(10, 6))
sns.histplot(data['statement_length'], bins=50, kde=True)
plt.title('Distribution of Statement lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
max_length = data['statement_length'].max()
plt.show()

#display basic statistics about statement lengths
print("\nStatistics of statement lengths:")
print(data['statement_length'].describe())

# prepare the feature matrix and target vector
X = data['statement']
y = data['status']

#encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)



# converts statement to numbers
tfidf = TfidfVectorizer(
    lowercase=True,                
    stop_words='english',          # gets rids of stopwords(dont contribute to meaning ie conjuctions and stuff)
    max_features=5000,             # too many unique words, do this to improve model performance    
    ngram_range=(1, 2)             # allow model to have 1 or 2 words in particular, ex: "happy" and "not happy"
    
) 


#TF-IDF is a statistical measure that evaluates how important a word is in a document relative to a collection of documents. 
# This is especially useful in machine learning models, as it gives more weight to important or frequently occurring words in a statement but reduces the weight of common words like "the," "and," etc.


# Term Frequency (TF): How frequently a word appears in a statement.
# Inverse Document Frequency (IDF): A measure of how rare a word is across all statements in the dataset. Rare words are given more weight.
# Apply the vectorizer to the 'statement' column to get the TF-IDF features
X_tfidf = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

#initialize the baseline model
model = MultinomialNB()

#train the model
model.fit(X_train, y_train)

#make predictions on the test set
y_pred = model.predict(X_test)

#evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

#confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)

#plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()






