# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Load the dataset from your local file using raw string to avoid path errors
df = pd.read_csv(r"C:\Users\hp\Downloads\archive (1)\spam.csv", encoding='latin-1')

# Preview the first few rows
print(df.head())

# Download stopwords
nltk.download('stopwords')

# Preprocessing: Drop unnecessary columns, rename columns for clarity
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert the label to a binary value: 'spam' = 1, 'ham' = 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Function to clean the text data
def clean_text(message):
    ps = PorterStemmer()
    message = message.lower()  # Convert to lowercase
    message = [char for char in message if char not in string.punctuation]  # Remove punctuation
    message = ''.join(message)
    words = [ps.stem(word) for word in message.split() if word not in stopwords.words('english')]  # Remove stopwords and stem words
    return ' '.join(words)

# Apply the cleaning function to the 'message' column
df['message_clean'] = df['message'].apply(clean_text)

# Split the dataset into training and testing sets
X = df['message_clean']
y = df['label']

# Vectorize the text data using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test on a new message
new_message = ["Congratulations! You've won a $1000 gift card. Call now!"]
new_message_clean = [clean_text(new_message[0])]
new_message_vector = cv.transform(new_message_clean)
prediction = model.predict(new_message_vector)
print("Prediction (1 = Spam, 0 = Ham):", prediction)
