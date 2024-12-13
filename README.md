# IMDB Sentiment Analysis Classification

## Overview
This project is a sentiment analysis classification model that determines whether a movie review is positive or negative. The model is trained on the IMDB dataset using a Simple Recurrent Neural Network (RNN) with an Embedding layer. It is implemented in Python with TensorFlow/Keras and deployed as a Streamlit web application for user interaction.

---

## Features
- Preprocessing of text data, including tokenization and padding.
- Simple RNN-based neural network model for binary sentiment classification.
- Streamlit app for real-time sentiment prediction.
- User input processing to classify movie reviews as "Positive" or "Negative."

---

## Dataset
The dataset used is the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/), which contains 50,000 movie reviews split equally into training and testing sets:
- **25,000 training reviews** (balanced between positive and negative sentiments).
- **25,000 testing reviews** (balanced between positive and negative sentiments).

---

## Project Structure
```
project-directory/
├── app.py                    # Streamlit app script
├── sentiment_model.h5        # Trained RNN model
├── requirements.txt          # Required Python libraries
├── README.md                 # Documentation
```

---

## Implementation

### Model Architecture
The model uses the following architecture:
- **Embedding Layer**: Converts words into dense vectors of fixed size.
- **Simple RNN Layer**: Processes sequences of word embeddings.
- **Dense Output Layer**: Outputs a single value (positive or negative sentiment) with a sigmoid activation function.


### Stramlit App Web UI APP:
### Features
- **Input Movie Review**: Users can input their own movie review text.
- **Prediction Output**: The app displays whether the review is positive or negative, along with the confidence score.

### Key Files
- `sentiment_model.h5`: Trained RNN model.
- `app.py`: Streamlit script for user interaction.

### Running the Streamlit App
Open the provided URL () in a web browser.
## File Requirements for Deployment
1. `sentiment_model.h5`: Trained model.
2. `app.py`: Streamlit app script.

---

## Requirements
- Python 3.7 or above
- TensorFlow
- Keras
- Streamlit
- Numpy
- Pickle


## Future Enhancements
- Integration  a more advanced model (e.g., LSTM or Transformer-based models like BERT).
- Deploy the app using cloud services such as AWS, Heroku, or Streamlit Community Cloud.
