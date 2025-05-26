# 🎬 IMDb Movie Review Sentiment Analysis using RNN

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red?style=flat-square&logo=keras)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)

A deep learning project built with TensorFlow and Keras to analyze the sentiment of IMDb movie reviews using a Recurrent Neural Network (Simple RNN). The model classifies reviews as either positive or negative, based on natural language text.

---

## 📌 Project Highlights

- ✅ End-to-end pipeline from data preprocessing to model saving
- 🧠 Built using a Simple RNN architecture for sequence learning
- 🗣️ Handles natural language movie reviews using word embeddings
- 💾 Uses Keras’ built-in IMDb dataset
- 🔁 Early stopping to prevent overfitting
- 💡 Great for beginners to understand sequence models

---

## 🧰 Tech Stack

- **Language**: Python
- **Libraries**: TensorFlow, Keras, NumPy, Pandas
- **Tools**: Jupyter Notebook, Matplotlib (optional for visualization)

---

## 📂 Project Structure

📦 imdb-sentiment-rnn
├── 📜 README.md
├── 📓 imdb_sentiment_rnn.ipynb
├── 📦 models/
│   └── simple_rnn_imdb.h5
├── 📄 requirements.txt
└── 📄 LICENSE

---

## 🧠 How It Works

1. **Dataset**: IMDb movie reviews with binary sentiment (positive/negative)
2. **Preprocessing**:
   - Restrict vocabulary to 10,000 most common words
   - Pad sequences to a fixed length of 500
3. **Model**:
   - Embedding → SimpleRNN → Dense(sigmoid)
   - Loss: Binary Crossentropy
   - Optimizer: Adam
4. **Training**:
   - EarlyStopping with validation loss monitoring
   - Training history tracked
5. **Saving**: Trained model saved as `.h5` file for future use

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure you have Python ≥3.9 and pip installed. Then install the required libraries:

```bash
pip install -r requirements.txt

▶️ Run the Project

Open the Jupyter notebook and run each cell in sequence:

jupyter notebook imdb_sentiment_rnn.ipynb

Or you can use Google Colab to run it online with no setup.

⸻

📈 Sample Results
	•	Training Accuracy: ~96–97%
	•	Validation Accuracy: ~85%
	•	Model: Simple RNN with 32 units and word embeddings
	•	Max review length: 500 words

🧪 The model performs well for basic sentiment classification tasks and provides an intuitive understanding of how RNNs process sequences in NLP.



⸻

🔐 License

This project is licensed under the MIT License - see the LICENSE file for details.

⸻

🤝 Connect with Me

Feel free to connect with me:
	•	🌐 LinkedIn: https://www.linkedin.com/in/navdeep-singh-398494b3/
	•	📫 Email: navdeepsinghdhangar@gmail.com
	•	💻 Portfolio: [Coming Soon]

⸻

⭐ If you like this project, consider giving it a star on GitHub!


