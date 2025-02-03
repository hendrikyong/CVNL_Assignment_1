# AI for Assistive Applications

## 📌 Project Overview:
1. **CNN (Convolutional Neural Network)** - Used for hand sign classification using the **ASL Alphabet Dataset**.  
2. **RNN (Recurrent Neural Network)** - Used for intent classification with the **SNIPS Dataset**.
3. **RNN (Recurrent Neural Network)** - Used for emotion & intent classification with the **Emotion Dataset**

## 🔹 About the Project  
🔹 **Project Name:** AI for Assistive Applications  
🔹 **Requirement 2.1:** ASL Hand Sign Recognition using CNN  
🔹 **Requirement 2.2:** Emotion & Intent Classification using RNN on SNIPS Dataset  
🔹 **Framework:** PyTorch  

## 👥 Team Members & Roles  
| Name  | Contribution (CNN) | Contribution (RNN) | ID |  
|---|-----|-----|----|  
| 🧑‍💻 Cheryl | Evaluation using metrics like accuracy, precision, recall, F1-score and confusion matrix | Design & Training of Model | S10258146H |  
| 🧑‍💻 Hendrik | Design & Training of Model, Implement Appropriate Preprocessing Techniques | Implement Appropriate Preprocessing Techniques | S10241624J |  
| 🧑‍💻 ZhiHeng | Implement mean and std function for calculation & Improve Model Performance | Improve Model Performance & Evaluation | S10241579@connect.np.edu.sg |  

## 📊 Dataset Information  
🖐 **ASL Hand Sign Recognition Dataset**  
📌 **Dataset Source:** [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
📌 **Classes:** 29 (A-Z, Space, Nothing, Delete)  

🗣 Intent Classification Dataset (SNIPS)  
📌 Dataset Source: [SNIPS Natural Language Dataset](https://github.com/sonos/nlu-benchmark)    
📌 Classes: 7 intents (Weather, Music, Restaurant, etc.)  

 🗣 Emotion Dataset (HuggingFace)
 📌Dataset Source: (https://huggingface.co/datasets/dair-ai/emotion)

## 🧠 CNN Model (Hand Sign Recognition)  
✔ 3 Convolutional Layers (with ReLU activation)  
✔ Max-Pooling Layers after each convolution  
✔ Dropout to prevent overfitting  
✔ Fully connected layers for classification  

## 🗣 RNN Model (Intent Classification)  
✔ Used LSTM (Long Short-Term Memory) for sequence modeling  
✔ Embedding layer applied for text feature extraction  
✔ Fully connected layers for classification  

## 🗣 RNN Model (Emotion Classification)  
✔ Used LSTM (Long Short-Term Memory) for sequence modeling  
✔ Embedding layer applied  
✔ Fully connected layers for classification  

## 📈 Training & Evaluation for Intent RNN 
📌 Optimizer: Adam (lr=1e-4)  
📌 Loss Function: CrossEntropyLoss  
📌 Regularization: Dropout (0.3)  

## 📈 Training & Evaluation for Emotion RNN 
📌 AdamW (lr=5e-3)
📌 CrossEntropyLoss
📌 Dropout (0.5)

✅ CNN Evaluation  
📌 Evaluated using Accuracy, Precision, Recall, F1-score  
📌 Used Confusion Matrix for error analysis  

✅ RNN Training for Intent RNN
📌 Tokenized dataset and trained using LSTM-based RNN  
📌 Used CrossEntropyLoss and AdamW Optimizer  

✅ RNN Training for Emotion RNN
📌 Tokenized dataset and trained using LSTM-based RNN  
📌 Used CrossEntropyLoss and AdamW Optimizer  

## 🏆 Results & Performance  
CNN Results (Hand Sign Recognition)  
📌 Final Model Performance:  
- Test Accuracy: %  


RNN Results (Intent Classification)  
📌 Final Model Performance:  
- Test Accuracy: 75.76%

RNN Results (Emotion Classification)  
📌 Final Model Performance:  
- Test Accuracy: 90.65%



## 🔗 References  
- PyTorch Documentation  
- Kaggle ASL Dataset  
- SNIPS NLU Dataset  
