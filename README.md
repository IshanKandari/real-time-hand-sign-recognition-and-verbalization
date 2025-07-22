# 🤟 Real-Time Hand Sign Recognition and Verbalization

This project enables real-time recognition of hand signs and converts them into speech and on-screen text, helping muted individuals communicate more effectively. It uses MediaPipe for extracting 3D hand landmarks, and a Random Forest classifier for gesture prediction.

## 📌 What We Built
1. Captures live hand signs using the webcam.  
2. Extracts 3D hand landmarks via MediaPipe.  
3. Saves the landmarks into a CSV file with user-defined labels.  
4. Trains a Random Forest classifier on the labeled data.  
5. Saves the trained model as a .pkl file (to skip retraining next time).  
6. Predicts hand signs in real time.  
7. Displays the prediction and converts it to speech using GTTS.

## 🧠 Technologies Used
- Python  
- OpenCV  
- MediaPipe  
- Pandas, NumPy  
- Scikit-learn  
- GTTS (Google Text-to-Speech)

## 🖥️ How to Run
1. Clone the repo  
2. Install requirements: `pip install -r requirements.txt`  
3. Run `collect_sign_data.py` to capture sign data  
4. Run `train_model.py` to train and save the model  
5. Run `real_time_recognizer.py` for live recognition and speech

## 📁 Files
- `collect_sign_data.py` → Captures hand landmarks  
- `train_model.py` → Trains and saves model  
- `real_time_recognizer.py` → Real-time sign prediction + TTS  
- `gesture_model.pkl` → Trained model  
- `isl_custom_landmarks.csv` → Collected hand data  
- `requirements.txt` → Dependencies  
- `README.md` → Project documentation

## 🚀 Future Ideas
- Add GUI for ease of use  
- Support more gestures  
- Multilingual speech output  
- Improve accuracy with deep learning

## 🎯 Tips
- Collect at least 30 samples per gesture for better results
- Perform the gesture clearly with one hand in front of webcam

## 🙏 Credits
Built using MediaPipe, GTTS, and Scikit-learn