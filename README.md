# Speech Emotion Recognition



Speech Emotion Recognition is a Python project that aims to classify emotions in speech recordings. This project utilizes machine learning techniques to identify emotions such as anger, happiness, sadness, and more from audio data. The process involves data collection, preprocessing, model training, evaluation, and a predictive system for real-time emotion recognition.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data Collection and Processing](#data-collection-and-processing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Predictive System](#predictive-system)


## Prerequisites

Before running the Loan Predictor script, make sure you have the following dependencies installed:

- Python (version >= 3.6)
- NumPy
- pandas
- librosa
- scikit-learn
- seaborn
- matplotlib
- IPython

## Getting Started

1. Clone this GitHub repository to your local machine:

   ```bash
   git clone https://github.com/Suhaas-Suran/Speech-Emotion-Recognition.git

## Usage

he Speech Emotion Recognition project can be used for the following tasks:

1. **Data Collection and Processing:** The project provides tools for loading, preprocessing, and exploring audio datasets containing emotional speech recordings.

2. **Model Training and Evaluation:** It trains machine learning models to recognize emotions in speech and evaluates their accuracy on test datasets.
   
3. **Predictive System:** You can use the trained models for real-time emotion recognition from live audio input.

Feel free to modify and extend the project for specific applications, such as emotion analysis in customer service calls, voice assistants, or sentiment analysis in multimedia content.
## Data Collection and Processing

The Speech Emotion Recognition project includes the following steps for data collection and processing:

1. **Importing Modules:**  The project imports essential Python libraries, including NumPy, pandas, librosa, scikit-learn, seaborn, matplotlib, and IPython.

2. **Loading the Dataset:** It provides functionality to load audio datasets containing emotional speech recordings. You can customize the dataset loading process to fit your specific dataset structure.

3. **Feature Extraction:** The project extracts audio features, such as Mel-frequency cepstral coefficients (MFCCs), to represent the speech data in a format suitable for model training.

4. **Data Split:** The dataset is split into training and testing sets to evaluate model performance.

5. **Data Visualization:** Visualization tools are included to gain insights into the distribution of emotions in the dataset.

You can customize this section to provide more details about your specific data preprocessing steps if needed.
## Model Training and Evaluation

The Speech Emotion Recognition project trains and evaluates machine learning models as follows:

1. **Feature Representation:** Extracted audio features are used as input for training models.

2. **Classifier Creation:** Various machine learning classifiers can be trained for emotion recognition, including decision trees, random forests, support vector machines (SVMs), and neural networks.

3. **Model Evaluation:** Model performance is assessed using metrics such as accuracy, precision, recall, and F1-score on a test dataset.

You can expand on this section to provide insights into the models used, their performance, and any fine-tuning or hyperparameter tuning.
## Predictive System

To create a predictive model for emotion classification based on the extracted MFCC features, you can use a machine learning classifier. Here we use RandomForestClassifier:

1. **Input Data:** Replace 'your_audio_file.wav' with the path to your audio file.

2. **Prediction:** The 'predict_emotion' function predicts the type of emotion.

Here's an example of how to use the predictive system:

# Example usage:
Replace 'your_audio_file.wav' with the path to your audio file
predicted_emotion = predict_emotion('your_audio_file.wav')
print(f"Predicted Emotion: {predicted_emotion}")
