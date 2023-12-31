#**Data sets Links**
[https://skylion007.github.io/OpenWebTextCorpus/](https://skylion007.github.io/OpenWebTextCorpus/)

#**Bigram Model**
### 1. Data Preparation:

#### Downloading the Text File:
```python
!wget https://github.com/Infatoshi/fcc-intro-to-llms/blob/main/wizard_of_oz.txt
```
Downloads a text file containing the text of "The Wizard of Oz" from a GitHub repository. This text file will serve as the dataset for training the bigram language model.

#### Importing Libraries:
```python
import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
Imports necessary PyTorch libraries for building and training neural networks. The `device` variable is set to 'cuda' if a GPU is available; otherwise, it defaults to 'cpu'.

#### Reading and Preprocessing the Text File:
```python
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocabulary_size = len(chars)
```
Reads the text from the file and extracts unique characters to form the vocabulary. The `chars` variable contains a sorted list of unique characters, and `vocabulary_size` is the total number of unique characters in the text.

#### Mapping Characters to Integers:
```python
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])
```
Creates mappings between characters and integers, allowing for easy conversion between characters and their corresponding integer representations. The `encode` function converts a string into a list of integers, and `decode` reverses the process.

#### Creating Tensor from Encoded Text:
```python
data = torch.tensor(encode(text), dtype=torch.long)
```
Converts the encoded text into a PyTorch tensor of type long. This tensor will be used as input to the bigram language model.

### 2. Model Definition:

#### Bigram Language Model Class:
```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # Generation method
        # ...
```
Defines a PyTorch neural network model for the bigram language model. The model has an embedding layer (`token_embedding_table`) to represent characters. The `forward` method calculates logits (unnormalized scores) and loss, while the `generate` method generates new sequences based on the learned patterns.

### 3. Training Setup:

#### Optimizer and Training Loop:
```python
model = BigramLanguageModel(vocabulary_size)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        print(f'step: {iter}, train loss: {train_loss}, val loss: {val_loss}')

    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
```
Initializes the bigram language model, moves it to the specified device, sets up the Adam optimizer, and runs a training loop. The training loop iterates through batches of data, computes logits and loss, performs backpropagation, and updates the model parameters.

### 4. Text Generation:

#### Generating Text with the Trained Model:
```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)
```
Generates new text sequences using the trained bigram model. It starts with an initial context (a tensor of zeros), predicts the next token in a loop, samples from the distribution of predicted probabilities, and appends the sampled token to the generated sequence. Finally, it decodes the generated sequence back into characters and prints the result.

### 5. Additional Functions:

#### Estimate Loss Function:
```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```
This function estimates the loss on both the training and validation sets. It is used during the training loop to monitor the model's performance.

#### Batch Creation Function:
```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```
This function generates a batch of input and target sequences for training or validation. It randomly selects starting indices and creates batches of input and target sequences.

### Summary:

In summary, this code implements a bigram language model using PyTorch, trains it on a text dataset, and demonstrates text generation based on the learned patterns. The model predicts the next character in a sequence given the preceding character, and this process is iteratively repeated to generate coherent and contextually relevant text. The training loop uses the Adam optimizer, and the model is evaluated using the estimate_loss function. The final generated text reflects the statistical patterns learned from the training data.

#**GPT model**
Certainly! Let's break down the code in more detail:

### Data Preprocessing (First Block):

1. **Import Libraries:**
   ```python
   import os
   import lzma
   from tqdm import tqdm
   ```

   - The code imports necessary libraries for working with files and displaying progress bars.

2. **Function to List ".xz" Files in a Directory:**
   ```python
   def xz_files_in_dir(directory):
       # ...
   ```

   - This function takes a directory path and returns a list of filenames with the ".xz" extension.

3. **Define Paths and Get File Lists:**
   ```python
   folder_path = "/content/openwebtext"
   output_file_train = "/content/output_train.txt"
   output_file_val = "/content/output_val.txt"
   vocab_file = "/content/vocab.txt"

   files = xz_files_in_dir(folder_path)
   total_files = len(files)
   ```

   - Paths for input files, output files, and the vocabulary file are specified.
   - The list of ".xz" files in the given directory is obtained.

4. **Calculate Split Indices and Process Files:**
   ```python
   split_index = int(total_files * 0.9)
   files_train = files[:split_index]
   files_val = files[split_index:]

   vocab = set()

   with open(output_file_train, "w", encoding="utf-8") as outfile:
       # ... (code for processing training files and updating vocabulary)

   with open(output_file_val, "w", encoding="utf-8") as outfile:
       # ... (code for processing validation files and updating vocabulary)

   with open(vocab_file, "w", encoding="utf-8") as vfile:
       # ... (code for writing vocabulary to file)
   ```

   - The code calculates the split index for training and validation files.
   - It processes the training and validation files separately, updating the vocabulary.

### Model Definition (Second Block):

1. **Import Libraries and Set Device:**
   ```python
   import torch
   import torch.nn as nn
   from torch.nn import functional as F
   import mmap
   import random
   import pickle
   import argparse
   ```

   - Necessary libraries for PyTorch, memory mapping, randomization, and argument parsing are imported.
   - The device (CPU or GPU) is set based on availability.

2. **Set Hyperparameters:**
   ```python
   batch_size = 32
   block_size = 128
   max_iters = 200
   learning_rate = 3e-4
   eval_iters = 100
   n_embd = 384
   n_head = 4
   n_layer = 4
   dropout = 0.2
   ```

   - Hyperparameters for the model training are set.

3. **Read Vocabulary and Create Mapping Dictionaries:**
   ```python
   chars = ""
   with open("/content/vocab.txt", 'r', encoding='utf-8') as f:
       text = f.read()
       chars = sorted(list(set(text)))

   vocab_size = len(chars)

   string_to_int = {ch: i for i, ch in enumerate(chars)}
   int_to_string = {i: ch for i, ch in enumerate(chars)}
   encode = lambda s: [string_to_int[c] for c in s]
   decode = lambda l: ''.join([int_to_string[i] for i in l])
   ```

   - The code reads the vocabulary file and creates dictionaries for character-to-index and index-to-character mappings.

4. **Function to Get Random Chunk of Text:**
   ```python
   def get_random_chunk(split):
       # ...
   ```

   - This function retrieves a random chunk of text from a specified file using memory mapping.

5. **Function to Get Batch of Data:**
   ```python
   def get_batch(split):
       # ...
   ```

   - This function obtains a batch of data for training or validation.

6. **Function to Estimate Loss:**
   ```python
   @torch.no_grad()
   def estimate_loss():
       # ...
   ```

   - This function estimates the loss on training and validation sets.

7. **Classes for Attention Head, MultiHead Attention, FeedForward, and Transformer Block:**
   ```python
   class Head(nn.Module):
       # ...

   class MultiHeadAttention(nn.Module):
       # ...

   class FeedFoward(nn.Module):
       # ...

   class Block(nn.Module):
       # ...
   ```

   - These classes define the components of the transformer model, including attention heads, multi-head attention, feedforward layers, and transformer blocks.

8. **GPTLanguageModel Class:**
   ```python
   class GPTLanguageModel(nn.Module):
       # ...
   ```

   - This class defines the GPT language model, incorporating the transformer components.

9. **Initialization and Weight Initialization:**
   ```python
   model = GPTLanguageModel(vocab_size)
   m = model.to(device)
   ```

   - An instance of the GPT model is created and moved to the specified device (GPU or CPU).
   - Weight initialization is performed.

### Model Training (Third Block):

1. **Create Optimizer:**
   ```python
   optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
   ```

   - An AdamW optimizer is created for training the model.

2. **Training Loop:**
   ```python
   for iter in range(max_iters):
       # ...
   ```

   - The training loop iterates through the specified number of epochs.

3. **Sample Batch of Data, Evaluate Loss, and Backpropagate:**
   ```python
   xb, yb = get_batch('train')
   logits, loss = model.forward(xb, yb)
   optimizer.zero_grad(set_to_none=True)
   loss.backward()
   optimizer.step()
   ```

   - A batch of data is sampled, and the model's loss is computed and backpropagated.

### Model Generation (Fourth Block):

1. **Save Trained Model:**
   ```python
   with open('model-01.pkl', 'wb') as f:
       pickle.dump(model, f)
   ```

   - The trained model is saved to a file using pickle.

2. **Generate Text Based on Prompt:**
   ```python
   prompt = 'Hello test test'
   context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
   generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
   print(generated_chars)
   ```

   - An example prompt is given, encoded, and used to generate new text using the trained GPT model.
   - The generated text is printed.

