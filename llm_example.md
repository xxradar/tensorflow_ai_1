## This script does the following:

- **Imports necessary modules**: TensorFlow and some specific layers and utilities for processing text.

- **Prepares the data**: It uses a simple string `text` as its dataset. In a real-world scenario, you would use a much larger and more complex dataset.

- **Tokenizes the text**: This converts the text into sequences of integers, where each integer represents a specific word.

- **Creates n-gram sequences**: This is important for training a language model, as it needs to learn the probability of a word given the previous words.

- **Pads the sequences**: This ensures that all sequences are of the same length.

- **Splits the data into predictors and labels**: The model will learn to predict the next word in a sequence.

- **Builds a Sequential model**: The model uses Embedding, GRU (a type of RNN), and Dense layers.

- **Compiles the model**: It uses categorical crossentropy as the loss function, suitable for multi-class classification tasks.

- **Trains the model**: The model is trained for 100 epochs. You might need to adjust the number of epochs based on your specific dataset.

```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample text (you can replace this with a larger dataset)
text = "Red lorry, yellow lorry, red lorry, yellow lorry."

# Tokenization and sequence generation
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
label = tf.keras.utils.to_categorical(label, num_classes=total_words)

# Model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(GRU(150, return_sequences=True))
model.add(GRU(100))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(predictors, label, epochs=100, verbose=1)
```
## This script does the following:
The `generate_text` function is updated to work with the latest TensorFlow API.<br>
It generates the predictions for the next word.<br>
The output is a list of probabilities for each word in the vocabulary.
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predictions = model.predict(token_list, verbose=0)
        predicted = np.argmax(predictions, axis=-1)[0]
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text



# Sample text (you can replace this with a larger dataset)
text = "Red lorry, yellow lorry, red lorry, yellow lorry."

# Tokenization and sequence generation
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
label = tf.keras.utils.to_categorical(label, num_classes=total_words)

# Model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(GRU(150, return_sequences=True))
model.add(GRU(100))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(predictors, label, epochs=100, verbose=1)

# Using the model
print(generate_text("Red lorry", 5, model, max_sequence_len))
```
