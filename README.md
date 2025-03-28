# Phishing Email Detection using Machine Learning

## Overview

This project looks to build a machine learning model capable of classifying emails as either "Safe Email" or "Phishing Email". It utilizes natural language processing techniques to extract features from the email text and trains a Logistic Regression model to perform the classification.

## Technologies Used

* Python 3
* pandas
* scikit-learn (for feature extraction, model training, and evaluation)
* nltk (optional, can be used for further text preprocessing)
* matplotlib and seaborn (optional, for data visualization)
* email (optional, for parsing email structures if needed for more advanced features)

## Setup Instructions

1.  **Clone the repository:**
    ```terminal
    git clone <YOUR_REPOSITORY_URL>
    cd phishing-email-detector
    ```
    (Replace `<YOUR_REPOSITORY_URL>` with the actual URL of your repository)

2.  **Create a virtual environment:**
    ```terminal
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**
    ```terminal
    source venv/bin/activate
    ```

4.  **Install the required dependencies:**
    ```terminal
    pip install pandas scikit-learn nltk matplotlib seaborn email
    Install the dependencies one by one (Recommended)
    ```

5.  **Download the dataset:**
    * Go to the Kaggle dataset page: [https://www.kaggle.com/datasets/rohit08/phishing-email-dataset](https://www.kaggle.com/datasets/rohit08/phishing-email-dataset)
    * Download the `Phishing_Email.csv` file.

6.  **Place the dataset in the `data` folder:**
    * Create a folder named `data` in the root of your project directory if it doesn't already exist.
    * Move the downloaded `Phishing_Email.csv` file into the `data` folder.

7.  **Run the main script:**
    ```terminal
    python3 Main.py
    ```

## Project Structure

phishing-email-detector/
├── data/
│   └── Phishing_Email.csv
├── Main.py
└── README.md
└── venv/
├── bin/
├── include/
└── lib/


## Results

The trained Logistic Regression model achieved the following evaluation metrics on the test dataset:

* **Accuracy:** Approximately 97.24%
* **Precision (for Phishing Email):** Approximately 95.62%
* **Recall (for Phishing Email):** Approximately 97.39%
* **F1-Score (for Phishing Email):** Approximately 96.50%

For a more detailed breakdown, please refer to the classification report printed by the script.

## Potential Improvements

* Explore other machine learning models (e.g., Naive Bayes, SVM, Random Forest).
* Implement more advanced feature engineering techniques (e.g., n-grams, word embeddings).
* Perform hyperparameter tuning to optimize model performance.
* Address the slight class imbalance in the dataset.
* Incorporate email header information for potentially richer features.

## Author
Esteban Jimenez Arias, ejimenez01