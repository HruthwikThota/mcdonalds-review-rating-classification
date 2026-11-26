# 🍔 McDonald's Customer Review Rating Prediction

## 1. Project Overview

This project focuses on predicting customer ratings based on written reviews of McDonald's restaurants using Natural Language Processing (NLP) and Deep Learning.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for text feature extraction and an **Artificial Neural Network (ANN)** for multi-class rating classification.

Optuna is used for hyperparameter tuning to identify a better-performing ANN configuration.

A Streamlit web application allows users to enter customer reviews and receive predicted ratings from 1 to 5 stars.

## 2. Project Objectives

- Analyze customer reviews using Natural Language Processing.
- Convert textual reviews into numerical features using TF-IDF.
- Develop an ANN model for five-class rating classification.
- Optimize ANN hyperparameters using Optuna.
- Evaluate model performance using classification metrics and a confusion matrix.
- Deploy the trained model through Streamlit Community Cloud.

## 3. Dataset Description

**Dataset:** McDonald's Customer Reviews

The dataset contains customer feedback collected from McDonald's restaurants.

- Original records: 33,396
- Empty reviews removed: 50
- Final records: 33,346
- Original columns: 10
- Input feature: `review`
- Target variable: `rating`

### Rating Classes

The original rating labels were converted into numerical values.

| Original Rating | Encoded Value |
|---|---:|
| 1 star | 0 |
| 2 stars | 1 |
| 3 stars | 2 |
| 4 stars | 3 |
| 5 stars | 4 |

The model predicts five rating classes.

## 4. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Regular Expressions | Text cleaning |
| Scikit-learn | TF-IDF and model evaluation |
| TensorFlow / Keras | ANN development |
| Optuna | Hyperparameter optimization |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| Joblib | Saving the TF-IDF vectorizer |
| Streamlit | Web application and deployment |
| GitHub | Version control and project hosting |

## 5. Project Workflow

The project follows these steps:

1. Dataset loading and exploration.
2. Data cleaning and preprocessing.
3. Removal of empty reviews.
4. Rating label encoding.
5. Text cleaning.
6. Train-test splitting.
7. TF-IDF feature extraction.
8. ANN model development.
9. Model training and validation.
10. Hyperparameter tuning using Optuna.
11. Model evaluation.
12. Saving the trained model and vectorizer.
13. Streamlit application development.
14. GitHub upload and cloud deployment.

## 6. Data Preprocessing

The following preprocessing operations were performed:

- Converted text into lowercase.
- Removed URLs.
- Removed HTML tags.
- Removed special characters and numbers.
- Removed extra spaces.
- Preserved apostrophes in words.
- Preserved important negation words such as "not", "no", and "never".
- Removed empty reviews.

The cleaned reviews were stored in a separate column named `cleaned_review`.

## 7. TF-IDF Feature Extraction

TF-IDF converts textual reviews into numerical feature vectors.

It assigns importance to words based on their frequency in a review and their distribution across the dataset.

**Configuration:**
- Maximum features: 2,000
- Input: Cleaned customer reviews
- Output: Numerical TF-IDF feature vectors

The vectorizer was fitted on the training data and then used to transform the test data.

The same fitted vectorizer is reused during deployment to process new customer reviews.

## 8. ANN Model Architecture

An Artificial Neural Network was developed using TensorFlow and Keras.

### Model Architecture

| Layer | Units | Activation |
|---|---:|---|
| Input | 2,000 features | - |
| Dense Layer 1 | 256 | ReLU |
| Dropout | - | 0.4139 |
| Dense Layer 2 | 256 | ReLU |
| Dropout | - | 0.4139 |
| Output Layer | 5 | Softmax |

### Model Configuration

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Batch Size: 32

The Softmax output layer produces probabilities for the five rating classes.

## 9. Hyperparameter Tuning Using Optuna

Optuna was used to search for suitable ANN hyperparameters.

The optimization process tested different configurations and selected the configuration with the highest validation accuracy.

### Tuned Hyperparameters

| Parameter | Best Value |
|---|---|
| Number of hidden layers | 2 |
| Units in Layer 1 | 256 |
| Units in Layer 2 | 256 |
| Dropout Rate | 0.4139 |
| Learning Rate | 0.0021966 |
| Optimizer | Adam |
| Batch Size | 32 |

The tuned model was trained using the selected configuration.

### Performance Comparison

| Model | Test Accuracy |
|---|---:|
| Initial TF-IDF + ANN | 66.25% |
| Optuna Tuned TF-IDF + ANN | 68.40% |

The tuned model improved test accuracy by approximately 1.14 percentage points.

## 10. Model Evaluation

The final tuned ANN model was evaluated using the test dataset.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Test Loss

### Final Test Results

| Metric | Result |
|---|---:|
| Test Accuracy | 68.40% |
| Macro Precision | 0.64 |
| Macro Recall | 0.61 |
| Macro F1-score | 0.62 |
| Weighted Precision | 0.67 |
| Weighted Recall | 0.68 |
| Weighted F1-score | 0.67 |
| Test Loss | 1.1335 |

### Classification Report

| Rating Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 1 Star | 0.73 | 0.87 | 0.79 |
| 2 Stars | 0.53 | 0.39 | 0.45 |
| 3 Stars | 0.58 | 0.49 | 0.54 |
| 4 Stars | 0.61 | 0.47 | 0.53 |
| 5 Stars | 0.73 | 0.81 | 0.77 |

The confusion matrix was used to examine correct predictions and misclassifications across all five rating classes.

## 11. Streamlit Web Application

The trained ANN model was integrated into a Streamlit application.

Users enter a customer review into the text input area and click the **Predict Rating** button.

### Application Workflow

1. Accept customer review text.
2. Clean the input text.
3. Transform the review using the saved TF-IDF vectorizer.
4. Pass the numerical features to the trained ANN model.
5. Predict the rating class.
6. Convert the predicted class into a 1-to-5-star rating.
7. Display prediction confidence and rating probabilities.

### Live Application

**Streamlit Application:**  
https://mcdonalds-review-rating-classification-udmbuzhzfajnbqwkzttv28.streamlit.app/

## 12. Sample Predictions

The deployed application was tested using three sample reviews.

| Customer Review | Predicted Rating | Confidence |
|---|---:|---:|
| The food was delicious and the service was excellent. | 5 Stars | 99.81% |
| Terrible service. The food was cold and the staff was rude. | 1 Star | 98.88% |
| The food was okay. Nothing special. | 3 Stars | 76.15% |

These are sample application predictions, not additional test-set evaluation results.

## 13. Repository Structure

```text
mcdonalds-review-rating-classification/
│
├── McDonald_s_Reviews.csv
│
├── McDonalds_Review_Rating_ANN.ipynb
│
├── app.py
│
├── final_ann_model.keras
│
├── tfidf_vectorizer.pkl
│
├── best_hyperparameters.json
│
├── requirements.txt
│
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `McDonald_s_Reviews.csv` | Original dataset |
| `McDonalds_Review_Rating_ANN.ipynb` | Complete model development and training notebook |
| `app.py` | Streamlit application code |
| `final_ann_model.keras` | Saved Optuna-tuned ANN model |
| `tfidf_vectorizer.pkl` | Saved fitted TF-IDF vectorizer |
| `best_hyperparameters.json` | Best Optuna hyperparameter configuration |
| `requirements.txt` | Required Python dependencies |
| `README.md` | Project documentation |

## 14. Deployment

The application was deployed using Streamlit Community Cloud.

Deployment process:

1. Created a GitHub repository.
2. Uploaded the notebook, application code, trained model, vectorizer, and dependencies.
3. Connected the GitHub repository to Streamlit Community Cloud.
4. Configured the application entry point as `app.py`.
5. Deployed the application.
6. Tested predictions using sample customer reviews.

## 15. Future Improvements

- Experiment with Word2Vec embeddings and ANN.
- Develop an LSTM-based rating classification model.
- Fine-tune a BERT model for rating prediction.
- Compare different NLP feature extraction techniques.
- Improve classification performance across underrepresented rating classes.
- Perform further model optimization and evaluation.
- Compare all models using accuracy, precision, recall, macro F1-score, and training time.

## 16. Conclusion

This project demonstrates an end-to-end NLP and Deep Learning workflow for customer review rating prediction.

It combines text preprocessing, TF-IDF feature extraction, ANN classification, Optuna hyperparameter tuning, model evaluation, and Streamlit deployment.

The deployed application accepts new customer reviews and predicts ratings from 1 to 5 stars using the saved trained model.

---

**Developed using Python, NLP, TensorFlow, Keras, Optuna, and Streamlit.**
