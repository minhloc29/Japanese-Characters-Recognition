# Japanese-Characters-Recognition

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

## Overview
Japanese Character Recognition is a deep learning project focused on recognizing ancient Japanese characters, such as Kanji and Hiragana, from historical texts and manuscripts. The project aims to develop a model that can accurately identify and classify these characters from images, even when they are degraded or distorted. By utilizing Convolutional Neural Networks (CNNs) and advanced preprocessing techniques, this system helps automate the digitization of valuable cultural heritage, enabling easier analysis and preservation of ancient Japanese writings.

## Dataset
This project uses datasets containing thousands of handwritten and printed Japanese pages from Kaggle.

### Source
- Segmentation task: [Segmenation Dataset](https://www.kaggle.com/datasets/minhlcnguyn/train-images)
- Classification task: [Classification Dataset](https://www.kaggle.com/datasets/minhlcnguyn/japanese-classification)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Streamlit

### Step
1. Clone the repository:
```bash
git clone https://github.com/user/repository.git](https://github.com/minhloc29/Japanese-Characters-Recognition.git
cd Japanese-Characters-Recognition
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the dataset and place it in the data/ directory.

## Usage
To run the Streamlit app for real-time predictions: 
```bash
streamlit run stream.py
```
### Making Predictions
Upload an image of a Japanese character, the app will display the recognized characters belong the original ones.

## Project Structure

japanese-character-recognition/ ├── data/ # Dataset directory │ ├── raw/ # Raw dataset files │ ├── processed/ # Preprocessed data ├── models/ # Saved model files ├── src/ # Source code │ ├── train.py # Script to train the model │ ├── evaluate.py # Script to evaluate the model │ ├── preprocess.py # Data preprocessing scripts │ ├── my_utils.py # Utility functions ├── tests/ # Unit test files │ ├── test_utils.py # Tests for utilities │ ├── test_model.py # Tests for model components ├── stream.py # Streamlit app for predictions ├── config.py # Configuration file for paths and settings ├── requirements.txt # Python dependencies ├── README.md # Project documentation └── LICENSE # License file

