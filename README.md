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



## SLM (Small Language Model)
Yeh project scratch se ek Small Language Model (SLM) banane par based hai. Iska main goal 50-60 million parameters ka model banana hai jo creative aur coherent text generate kar sake. Is project mein PyTorch library use ki gayi hai aur training ke liye "TinyStories" dataset ka use kiya gaya hai.

## Dataset Import (Step 1)
TinyStories dataset, jo GPT-3.5 aur GPT-4 dwara 3-4 saal ke bachchon ke liye banayi gayi hai, HuggingFace se import kiya gaya hai.

## Data Preprocessing (Steps 2 & 3)
Tokenization: Dataset ko tiktoken library se tokenIDs mein tokenize kiya gaya hai.

* Storage: Tokenized data ko efficient computation ke liye RAM ke bajay disk par train.bin aur validation.bin files mein store kiya gaya hai.

* Batching: Model training ke liye input-output batches banane ke liye ek get_batch function banaya gaya hai.

## Model Architecture (Step 4)
Model architecture mein yeh components shamil hain:

* LayerNorm

* CausalSelfAttention

* MLP (Multi-Layer Perceptron)

* Block (Transformer block)

* GPT (Main model)

* Model ko configure karne ke liye GPTConfig ka upyog kiya gaya hai, jismein vocab_size, block_size, n_layer, n_head, aur n_embd jaise parameters define kiye gaye hain.

## Training and Evaluation (Steps 5-9)
* Loss Function: Model ke training aur validation loss ko calculate karne ke liye estimate_loss function banaya gaya hai.

* Configuration: Training ke hyperparameters jaise learning_rate, max_iters, batch_size, gradient_accumulation_steps, aur optimizer define kiye gaye hain.

* Scheduler: Training ko smooth aur stable banane ke liye LinearLR aur CosineAnnealingLR ka use karte hue ek learning rate scheduler implement kiya gaya hai.

* Loss Plotting: Model ke training aur validation loss ko visualize karne ke liye ek plot banaya gaya hai.

## Model Inference (Step 10)
Training ke baad, best model ko load karke usse naya text generate kiya gaya hai. Model ki text generation capability ko generate function ka upyog karke Once upon a time there was a pumpkin. aur A little girl went to the woods jaise prompts par test kiya gaya hai.






## 🚀 Results and Inference
The model was trained for 20,000 iterations. Both the training and validation losses consistently decreased, showing that the model learned successfully.

After training, the model was given an initial phrase ("Once upon a time there was a pumpkin") to generate new stories. Instead of producing random words, the model generated grammatically correct and coherent sentences, proving that it has learned language and storytelling.








