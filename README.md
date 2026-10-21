# Traffic Sign Recognition System

## Project Overview
The Traffic Sign Recognition System is an AI/ML project that uses a fine-tuned MobileNetV2 convolutional neural network (CNN) to classify traffic sign images into three categories: Cautionary, Informatory, and Mandatory.

## Features
- Upload a traffic sign image through a Streamlit web interface.
- Predict the traffic sign category.
- Display the model's confidence score.
- Show class-wise prediction scores.
- Evaluate model performance using a confusion matrix and classification report.

## Technologies Used
- Python
- TensorFlow and Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow
- Pandas, Matplotlib and Seaborn for analysis and visualizations

## Dataset
The project uses the Indian traffic sign image dataset downloaded from Zenodo.

Source: https://zenodo.org/records/17300841

The dataset images were labelled into three categories: Cautionary, Informatory and Mandatory. Refer to the original dataset source for its applicable license and usage conditions.

## Model
The project uses a fine-tuned MobileNetV2 model. Input images are resized to 160 × 160 pixels before prediction.

The model file is expected at:

`model/traffic_sign_mobilenetv2.keras`

## Run the Streamlit Application

1. Install Python and the dependencies.
2. Open a terminal in the project directory.
3. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

4. Start the application:

```bash
python -m streamlit run app.py
```

5. Open the local URL displayed in the terminal and upload a traffic sign image.

## Project Structure

```text
Traffic_Sign_Recognition_System/
├── app.py
├── requirements.txt
├── README.md
├── TeamC_03_PS15.ipynb
├── model/
│   └── traffic_sign_mobilenetv2.keras
├── dataset/
└── results/
```

## Limitations
The model predicts one of three broad traffic sign categories. It is not designed to identify every individual sign or replace a real-world traffic safety system.

## Team Project
Problem Statement XV — Traffic Sign Recognition System.