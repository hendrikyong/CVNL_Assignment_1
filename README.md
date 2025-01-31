# AI for Assistive Applications

## 📌 Project Overview:
1️⃣ **CNN (Convolutional Neural Network)** - Used for hand sign classification using the **ASL Alphabet Dataset**.  
2️⃣ **RNN (Recurrent Neural Network)** - Used for emotion & intent classification with the **SNIPS Dataset**.  

## 🔹 About the Project  
🔹 **Project Name:** AI for Assistive Applications  
🔹 **Requirement 2.1:** ASL Hand Sign Recognition using CNN  
🔹 **Requirement 2.2:** Emotion & Intent Classification using RNN on SNIPS Dataset  
🔹 **Framework:** PyTorch  

## 👥 Team Members & Roles  
| Name  | Contribution (CNN) | Contribution (RNN) | ID |  
|---|-----|-----|----|  
| 🧑‍💻 Cheryl | Evaluation | Design & Training of Model | S102 |  
| 🧑‍💻 Hendrik | Design & Training of Model, Implement Appropriate Preprocessing Techniques | Implement Appropriate Preprocessing Techniques | S10241624J |  
| 🧑‍💻 ZhiHeng | Implement mean and std function for calculation & Improve Model Performance | Improve Model Performance & Evaluation | S102 |  

## ⚙️ Installation & Setup  


## 📊 Dataset Information  
🖐 **ASL Hand Sign Recognition Dataset**  
📌 **Dataset Source:** [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
📌 **Classes:** 29 (A-Z, Space, Nothing, Delete)  

🗣 Emotion & Intent Classification Dataset (SNIPS)  
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

## 🧠 RNN Model (Emotion & Intent Classification)  
✔ Used LSTM (Long Short-Term Memory) for sequence modeling  
✔ Embedding layer applied for text feature extraction  
✔ Fully connected layers for classification  

## 📈 Training & Evaluation  
📌 Optimizer: Adam (lr=1e-4)  
📌 Loss Function: CrossEntropyLoss  
📌 Regularization: Dropout (0.6)  

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
