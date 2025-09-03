# 📝 Small Language Model (SLM) From Scratch
This project focuses on building a small language model (SLM) completely from scratch, from the ground up. It includes both training and inference. The model is trained on a specific dataset to enable it to generate coherent English stories.

## 🌟 Project Goal

We've created a 15-million-parameter model. The main goal is for the model to understand language, specifically stories written for 3-4 year-olds, and then generate new stories on its own. This proves that high-quality models can be built on small, well-curated datasets instead of massive ones.

## 🛠️ Technologies and Components
* Pytorch: Used for the model architecture, training, and building custom components.

* Google Colab (A100 GPU): For computational efficiency and fast training.

* Ticktoken Library: To tokenize text into GPT-2-style subwords.

* Datasets Library (Hugging Face): To load the Tiny Stories dataset.

* Numpy: For data handling and memory-mapped arrays.

## 📂 Dataset
This project uses the Tiny Stories dataset, which contains over 2 million small stories written by GPT-4 for children aged 3-4.

## ⚙️ How It Works
Data Preprocessing (Tokenization): The text is converted into numbers (token IDs). Byte Pair Encoding (BPE) is used for this. All token IDs are saved to train.bin and validation.bin files on disk to prevent RAM overload.

* Creating Input/Output Pairs: To train the model to predict the next token, input-output pairs are created where the output is always one token ahead of the input. This helps the model learn next-token prediction.

* Model Architecture: We've built a Transformer architecture with the following components:

* Token and Positional Embeddings: To convert words into vectors with positional information.

* Layer Normalization: To stabilize training.

* Causal Self-Attention: To ensure each token only attends to preceding tokens. This helps the model understand context.

* Feed-Forward Network: To make the model more expressive.

* Shortcut Connections: To provide an alternative path for gradients, preventing the vanishing gradient problem.

## Training Loop:

* Loss Function: We use Negative Log Likelihood or Cross-Entropy Loss to compare the model's predictions with the actual targets.

* Optimizer: The AdamW optimizer is used.

* Learning Rate Schedule: A linear warm-up followed by a cosine decay learning rate is used to smooth out training.

* Gradient Accumulation: To handle large batch sizes that can't fit on the GPU at once.

* Mixed Precision Training: To speed up computations by using lower-precision (float16) operations where safe.

## 🚀 Results and Inference
The model was trained for 20,000 iterations. Both the training and validation losses consistently decreased, showing that the model learned successfully.

After training, the model was given an initial phrase ("Once upon a time there was a pumpkin") to generate new stories. Instead of producing random words, the model generated grammatically correct and coherent sentences, proving that it has learned language and storytelling.















