# 🖐 ASL Hand Sign & Intent Classification using CNN & RNN

## 📌 Project Overview:
This project implements two deep learning models:  
1️⃣ **CNN (Convolutional Neural Network)** - Used for hand sign classification using the **ASL Alphabet Dataset**.  
2️⃣ **RNN (Recurrent Neural Network)** - Used for intent classification with the **SNIPS Dataset**.  


## 📜 Table of Contents  
- 🔹 About the Project  
- 👥 Team Members & Roles  
- 📂 Project Structure  
- ⚙️ Installation & Setup  
- 📊 Dataset Information  
- 🧠 CNN Model (Hand Sign Recognition)  
- 🧠 RNN Model (Intent Classification)  
- 📈 Training & Evaluation  
- 🏆 Results & Performance  
- 🔗 References  


## 🔹 About the Project  
🔹 **Project Name:** ASL Hand Sign & Intent Classification  
🔹 **Requirement 2.1:** ASL Hand Sign Recognition using CNN  
🔹 **Requirement 2.2:** Intent Classification using RNN on SNIPS Dataset  
🔹 **Framework:** PyTorch  
🔹 **Techniques Used:** Data Augmentation, CNN, RNN, Hyperparameter Tuning, Early Stopping  
🔹 **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  


## 👥 Team Members & Roles  
| Name  | Contribution**(CNN)** | Contribution**(RNN)** | ID |  
|---|-----|-----|----|  
| 🧑‍💻 Cheryl | Evaluation & Confusion Matrix | Design & Training of Model | S102 |  
| 🧑‍💻 Hendrik | Design & Training of Model | Imeplement Preprocessing Techniques | S102 |  
| 🧑‍💻 ZhiHeng | Implement Preprocessing Techniques & Improve Model Performance | Improve Model Performance, Evaluation & Confusion Matrix | S102 |  


## 📂 Project Structure  
📂 ASL-Intent-Classification
│── 📁 data/                # Dataset files (ASL & SNIPS)  
│── 📁 models/              # Trained models (CNN & RNN)  
│── 📄 README.md            # Project documentation  
│── 📄 requirements.txt     # Dependencies and libraries  
│── 📄 CNN.ipynb            # CNN Training script  
│── 📄 RNN.ipynb            # RNN Training script  


## ⚙️ Installation & Setup  


## 📊 Dataset Information  
🖐 ASL Hand Sign Recognition Dataset  
📌 Dataset Source: Kaggle ASL Alphabet Dataset  
📌 Classes: 29 (A-Z, Space, Nothing, Delete)  
📌 Preprocessing: 
- Resized images to 128x128  
- Applied data augmentation (flipping, rotation, color jittering)
- Normalised images using dataset-specific **mean and std**


## 🗣 Intent Classification Dataset (SNIPS)  
📌 Dataset Source: SNIPS Natural Language Dataset  
📌 Classes: 7 intents (Weather, Music, Restaurant, etc.)  
📌 Preprocessing:
- Tokenization using NLTK  
- Embedded text with word embeddings  
- Used LSTM-based RNN for intent classification  


## 🧠 CNN Model (Hand Sign Recognition)  
✔ 3 Convolutional Layers (with ReLU activation)  
✔ Max-Pooling Layers after each convolution  
✔ Dropout (0.6) to prevent overfitting  
✔ Fully connected layers for classification  


## 🧠 RNN Model (Intent Classification)  
✔ Used LSTM (Long Short-Term Memory) for sequence modeling  
✔ Embedding layer applied for text feature extraction  
✔ Fully connected layers for classification  


## 📈 Training & Evaluation  
📌 Optimizer: Adam (lr=1e-4)  
📌 Loss Function: CrossEntropyLoss  
📌 Regularization: Dropout (0.6)  
📌 Learning Rate Scheduler: StepLR (step_size=5, gamma=0.1)  
📌 Early Stopping: Implemented to stop training when validation loss stops improving  

✅ CNN Evaluation  
📌 Evaluated using Accuracy, Precision, Recall, F1-score  
📌 Used Confusion Matrix for error analysis  

✅ RNN Training  
📌 Tokenized dataset and trained using LSTM-based RNN  
📌 Used CrossEntropyLoss and Adam Optimizer  


## 🏆 Results & Performance  
CNN Results (Hand Sign Recognition)  
📌 Final Model Performance:  
- Test Accuracy: %  
- Confusion Matrix:  
- Loss Graph:

RNN Results (Intent Classification)  
📌 Final Model Performance:  
- Test Accuracy: %  
- Confusion Matrix:  
- Loss Graph:  


## 🔗 References  
- PyTorch Documentation  
- Kaggle ASL Dataset  
- SNIPS NLU Dataset  
