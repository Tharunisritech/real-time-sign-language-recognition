\# Real-Time Sign Language Recognition Using Deep Learning



\## Project Description



This project develops a deep learning-based system for recognizing American Sign Language (ASL) alphabet hand gestures.



The system uses a Convolutional Neural Network (CNN) trained on the ASL Alphabet Dataset.



\## Objective



The objective is to recognize hand gestures representing ASL alphabet classes and predict the corresponding class.



\## Dataset



Dataset: ASL Alphabet Dataset



Source:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet



The dataset contains alphabet and additional gesture classes such as `del`, `nothing`, and `space`.



\## Technologies Used



\- Python

\- TensorFlow

\- Keras

\- OpenCV

\- MediaPipe

\- NumPy

\- Pandas

\- Scikit-learn

\- Matplotlib



\## Model



A Convolutional Neural Network (CNN) was trained using the ASL Alphabet Dataset.



The trained model is stored in:



`model/sign\_language\_model.keras`



\## Results



Training Accuracy: 98.18%



Validation Accuracy: 99.92%



Test Accuracy: 100% on the provided 28-image test set.



\## Project Structure



```text

Sign\_Language\_Project/

│

├── model/

│   ├── sign\_language\_model.keras

│   └── hand\_landmarker.task

│

├── src/

│   ├── train.py

│   ├── test.py

│   ├── webcam.py

│   └── train\_augmented.py

│

├── requirements.txt

└── README.md

