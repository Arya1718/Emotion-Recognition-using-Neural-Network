# Emotion Recognition with CNN and TensorFlow

This project implements an emotion recognition system using a Convolutional Neural Network (CNN) trained on a dataset of facial expressions. The model is capable of predicting six different emotions from images: **Happy**, **Anger**, **Pain**, **Disgust**, **Fear**, and **Sad**. The project is further extended to a web-based interface using **Streamlit**, where users can upload images or take pictures to detect the emotion displayed in them.

## Table of Contents

- [Features](#features)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Augmentation**: The project uses `ImageDataGenerator` for on-the-fly data augmentation (shear, zoom, and horizontal flip).
- **CNN Model**: A 4-layer CNN model that classifies images into six emotion categories.
- **Model Training**: The model is trained with 80% of the dataset and validated with the remaining 20%.
- **Streamlit Interface**: Allows users to upload images or use their camera to predict the emotion displayed.
- **Progress Bar**: A visual progress bar simulates the processing time of the model's prediction in the web app.

## Model Architecture

The CNN architecture is as follows:

1. **Convolutional Layer 1**: 32 filters of size 3x3, ReLU activation, followed by MaxPooling.
2. **Convolutional Layer 2**: 64 filters of size 3x3, ReLU activation, followed by MaxPooling.
3. **Convolutional Layer 3**: 128 filters of size 3x3, ReLU activation, followed by MaxPooling.
4. **Convolutional Layer 4**: 256 filters of size 3x3, ReLU activation, followed by MaxPooling.
5. **Dense Layers**: A Dense layer with 256 units and ReLU, followed by a Dropout of 0.5. Another Dense layer with 64 units and ReLU.
6. **Output Layer**: Dense layer with 6 units (for the 6 emotion classes) and a softmax activation.

## Data Preprocessing

The dataset is split into training and validation sets (80% and 20% respectively). The images are resized to `64x64` and normalized (pixel values scaled between 0 and 1).

- **Augmentations**: The dataset is augmented with random shear, zoom, and horizontal flip to improve generalization.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Arya1718/Emotion-Recognition-using-Neural-Network.git
    cd emotion-recognition
    ```

2. **Install the required libraries**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset** and place it in a folder named `Emotions`. The dataset should contain subfolders named after the emotion categories (e.g., `happy`, `anger`, `pain`, etc.).

4. **Run the application**:

    To train the model:

    ```bash
    python train_model.py
    ```

    To launch the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Usage

- **Training**: The CNN model is trained on the dataset provided in the `Emotions` directory. The model is saved as `emotion_recognition_model.h5` after training.
- **Predicting Emotions**: The `predict_emotion()` function in the app uses the trained model to predict the emotion from a given image.
- **Streamlit Interface**: Use the web-based interface to upload an image or capture one with your camera to get a real-time prediction.

## Dataset

The dataset used for this project should consist of labeled images for different emotions such as **Happy**, **Anger**, **Pain**, **Disgust**, **Fear**, and **Sad**. Ensure that the images are categorized into separate directories for each emotion.

The dataset should follow this structure:

```
Emotions/
├── happy/
├── anger/
├── pain/
├── disgust/
├── fear/
└── sad/
```

## Results

The CNN model achieves **87.5% accuracy** on the validation dataset after training for **100 epochs**. The model performance can be further improved by tuning the hyperparameters or using a larger dataset.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug fixes, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.