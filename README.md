# Arabic → English Neural Machine Translation (Transformer, PyTorch)
## Final Project – Pattern Recognition (PR)

This repository contains a **Neural Machine Translation (NMT)** project  
that translates **Arabic → English** using a **Transformer encoder–decoder** architecture implemented from scratch in **PyTorch**.

The work was done as a **final project** for the **Pattern Recognition (PR)** course.

---

## 🎯 Project Goal

- Build and train a **seq2seq** model to translate Arabic sentences into English.
- Implement and understand the **Transformer** architecture (encoder–decoder).
- Apply **Pattern Recognition** concepts on a real-world **Arabic–English parallel dataset**.
- Track and visualize **training and validation performance**.

---

## 📂 Dataset

- **File:** `ara_.txt`  
- **Type:** Parallel corpus (Arabic sentence ⟷ English sentence)  
- **Usage:**  
  - Used for training, validation, and testing.  
  - Preprocessed for tokenization, cleaning, and vocabulary building.

> ⚠️ The dataset file (`ara_.txt`) is not included here if it is private.  
> Make sure to place it in the correct path before running the notebook.

---

## 🧠 Model – Transformer Encoder–Decoder

The notebook implements a **Transformer-based NMT model**:

- Architecture: **Encoder–Decoder Transformer** (`nn.Transformer`)
- Framework: **PyTorch**
- Input: Tokenized **Arabic** sentences
- Output: Generated **English** translations

Key features:

- Custom **tokenization & vocabulary**
- Positional encodings
- Padding & attention masks
- Teacher forcing during training

---

## 📝 Notebook Outline

The main Jupyter Notebook (`Project_PR.ipynb`) is organized as follows:

1. **Load and inspect the dataset**  
2. **Text preprocessing**  
   - Cleaning
   - Tokenization (Arabic & English)
   - Building vocabularies  
3. **Numericalization**  
   - Converting tokens to indices  
   - Creating PyTorch `Dataset` and `DataLoader`  
4. **Build the Transformer model**  
   - Encoder–decoder using `nn.Transformer`  
5. **Training loop**  
   - Forward pass  
   - Loss calculation (Cross-Entropy)  
   - Backpropagation and optimizer step  
6. **Metrics & Visualizations**  
   - Training loss and accuracy  
   - Validation loss and accuracy  
   - 2 plots: **Loss vs Epochs** & **Accuracy vs Epochs**  
7. **Inference**  
   - Translate new Arabic sentences  
   - Greedy decoding / step-by-step generation  
8. **Qualitative Evaluation**  
   - Show random validation examples:  
     - Input (Arabic)  
     - Predicted output (English)  
     - Ground truth (English)  
9. **Optional: BLEU score**  
   - Compute a simple **BLEU** metric on a small validation subset  

---

## 🛠️ Technologies & Dependencies

Main stack:

- **Python 3.x**
- **PyTorch**
- **Jupyter Notebook**
- `matplotlib` / `seaborn` (for plots)
- `torchtext` or custom preprocessing (depending on implementation)
- `nltk` or `sacrebleu` (optional for BLEU)

Example installation:

``bash
pip install torch torchvision torchaudio
pip install matplotlib nltk sacrebleu
🚀 How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/<your-username>/arabic-english-transformer-nmt.git
cd arabic-english-transformer-nmt
Place the dataset

Put ara_.txt in the expected folder (for example: data/ara_.txt).

Adjust the path inside the notebook if needed.

Create and activate a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt   # if you create one
or manually install as listed above.

Run the notebook:

bash
Copy code
jupyter notebook Project_PR.ipynb
📊 Training & Evaluation
During training, the notebook:

Logs training loss & accuracy per epoch

Evaluates on a validation set

Plots:

Loss vs Epochs

Accuracy vs Epochs

Optionally:

Computes a simple BLEU score on a validation subset.

Displays sample translations to visually inspect quality.

🧪 Inference – Translating New Sentences
The notebook includes cells for inference:

You can input a custom Arabic sentence

The model outputs an English translation

Useful for demo and grading purposes

Example (inside the notebook):

python
Copy code
arabic_sentence = "اكتب الجملة العربية هنا"
translation = translate_sentence(arabic_sentence, model, src_vocab, trg_vocab, device)
print(translation)
📚 Educational Context
This project was developed as a final project for the:

Course: Pattern Recognition (PR)

Topic: Sequence modeling and Neural Machine Translation

Focus: Applying PR ideas using deep learning and Transformers on Arabic–English data.

📌 Notes
This is a teaching/demo project, not a production system.

Translation quality depends heavily on:

Dataset quality and size

Hyperparameters

Training time and hardware

✍️ Author
Student: Ahmed Sameh

Course: Pattern Recognition (PR)

Project: Arabic → English Neural Machine Translation (Transformer, PyTorch)
