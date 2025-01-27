# Natural Language Processing with Disaster Tweets

## Overview
This project is part of a Kaggle competition that involves classifying tweets as either related to real disasters or not. 
The dataset consists of labeled tweets where `1` indicates a disaster-related tweet and `0` indicates otherwise.

The goal is to build a Natural Language Processing (NLP) model that accurately classifies these tweets.

## Dataset
The dataset consists of two CSV files:
- **train.csv**: Contains labeled tweets for training the model.
- **test.csv**: Contains unlabeled tweets for prediction.

### Features:
- `id`: Unique identifier for each tweet.
- `text`: The tweet text.
- `target`: (Only in train.csv) 1 for disaster-related tweets, 0 for others.

## Project Steps

1. **Data Preprocessing:**
   - Removing URLs, special characters, and stopwords.
   - Tokenizing and padding sequences for deep learning models.

2. **Exploratory Data Analysis (EDA):**
   - Understanding dataset distribution.
   - Visualizing word clouds and class distributions.

3. **Model Building & Training:**
   - Using a Bidirectional LSTM model with word embeddings.
   - Training on the cleaned text data.

4. **Evaluation & Hyperparameter Tuning:**
   - Experimenting with different architectures and parameters.
   - Measuring accuracy and classification metrics.

5. **Generating Kaggle Submission:**
   - Predicting labels for test data.
   - Creating a submission file for Kaggle.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/NLP-Disaster-Tweets.git
   cd NLP-Disaster-Tweets
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data) and place `train.csv` and `test.csv` in the project directory.

4. Run the Jupyter Notebook:
   ```sh
   jupyter notebook NLP_Disaster_Tweets.ipynb
   ```

## Model Architecture
The model is built using **Bidirectional LSTM** with an embedding layer, dropout layers, and a fully connected output layer.

**Architecture:**
- Embedding Layer (word representations)
- Spatial Dropout
- Bidirectional LSTM
- Dense Output Layer with Sigmoid Activation

## Results
- Model Performance: Achieved reasonable accuracy in classifying tweets.
- Submission File: `submission.csv` generated for Kaggle.

## Deliverables
- **Jupyter Notebook:** Complete implementation of data processing, model training, and predictions.
- **GitHub Repository:** Contains all project files, including code and documentation.
- **Kaggle Leaderboard Screenshot:** Showing the model's rank after submission.

## References
- [Kaggle Competition](https://www.kaggle.com/competitions/nlp-getting-started)
- TensorFlow, Keras, Scikit-learn, NLTK

---
📌 **GitHub Repository:** [Link to Repository](#)  
📌 **Kaggle Leaderboard:** Will be updated after final evaluation.

