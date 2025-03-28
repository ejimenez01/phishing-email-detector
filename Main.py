import pandas as pd

# Define the path to your dataset
data_path = 'data/Phishing_Email.csv'

try:
    # Load the dataset
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully!")
    print("\nFirst 10 rows of the dataset:")
    print(df.head(10))

    print("\nInformation about the dataset:")
    print(df.info())

    print("\nSummary statistics of numerical columns:")
    print(df.describe())

    print("\nChecking for missing values:")
    print(df.isnull().sum())

    # Let's look at the distribution of the target variable (assuming a column indicates phishing)
    # You'll need to identify the name of the column that labels emails as phishing or not
    if 'Email Type' in df.columns:  # Example column name, adjust if yours is different
        print("\nDistribution of 'Email Type':")
        print(df['Email Type'].value_counts())
    elif 'is_phishing' in df.columns: # Another common name
        print("\nDistribution of 'is_phishing':")
        print(df['is_phishing'].value_counts())
    else:
        print("\nWarning: Could not find a standard target variable column name. Inspect df.columns.")

    # You might also want to look at the content of the emails
    if 'Email Text' in df.columns: # Example column name, adjust if yours is different
        print("\nSample of Email Text (first 5 emails):")
        print(df['Email Text'].head())
    elif 'text' in df.columns: # Another common name
        print("\nSample of Email Text (first 5 emails):")
        print(df['text'].head())
    else:
        print("\nWarning: Could not find a standard email text column name. Inspect df.columns.")

except FileNotFoundError:
    print(f"Error: Dataset not found at {data_path}. Make sure the file is in the correct location.")
except Exception as e:
    print(f"An error occurred: {e}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Handle missing values in 'Email Text' by filling with an empty string
df['Email Text'].fillna('', inplace=True)

# Separate features (X) and target (y)
X = df['Email Text']
y = df['Email Type']

# Convert target variable to numerical values (0 for Safe Email, 1 for Phishing Email)
y = y.map({'Safe Email': 0, 'Phishing Email': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("\nShape of X_train_tfidf:", X_train_tfidf.shape)
print("Shape of X_test_tfidf:", X_test_tfidf.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

from sklearn.linear_model import LogisticRegression

# Initialize a Logistic Regression model
logistic_regression_model = LogisticRegression(random_state=42)

# Train the model using the training data
logistic_regression_model.fit(X_train_tfidf, y_train)

print("\nLogistic Regression model trained successfully!")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Make predictions on the test data
y_pred = logistic_regression_model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Print a more detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe Email', 'Phishing Email']))
