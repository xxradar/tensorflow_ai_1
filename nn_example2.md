## Code
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
text = f"""
Artificial Intelligence (AI) is a broad branch of computer science concerned with building smart machines capable of performing tasks that typically require human intelligence. AI is an interdisciplinary science with multiple approaches, but advancements in machine learning and deep learning are creating a paradigm shift in virtually every sector of the tech industry.

AI systems are powered by algorithms, using techniques such as machine learning, deep learning, and rules-based systems. Machine learning algorithms feed computer data to AI systems, using statistical techniques to enable AI systems to learn. Through machine learning, AI systems get progressively better at tasks without having to be specifically programmed for them. Deep learning, a subset of machine learning, structures algorithms in layers to create an "artificial neural network" that can learn and make intelligent decisions on its own.

AI is a significant part of the technology industry. Research associated with AI is highly technical and specialized. The core problems of artificial intelligence include programming computers for certain traits such as knowledge, reasoning, problem-solving, perception, learning, planning, and the ability to manipulate and move objects. Long-term goals of AI research include achieving Creativity, Social Intelligence, and General (Human Level) Intelligence.

AI has been used to develop and advance numerous fields and industries, including finance, healthcare, education, transportation, and more. In finance, AI technologies can be used to identify which transactions are likely to be fraudulent, adopt fast and accurate credit scoring, as well as automate manually intense data management tasks. AI in healthcare is being used for dosing drugs and different treatment in patients, and for surgical procedures in the operating room.

The use of AI in education makes a system that is more adaptable to the needs of students. Virtual tutors and personalized learning environments are being developed to cater to the individual needs of students. AI is also being used in the transportation industry to manage traffic, predict flight delays, and make ocean shipping safer and more efficient.

AI is also used in the daily operations of many companies. It is used in automating tasks for low-level employees to higher-ranking officials. AI technologies help in scheduling trains, assessing business risk, predicting maintenance, and improving energy efficiency, among many other uses.

AI is an integral part of the future of technology. It is being used to help solve many big and small problems, from cancer to customer experience. AI is expected to become a part of daily life in many different ways. In the future, AI will become a little more sophisticated and might be able to perform more complex tasks. AI is also expected to be used in the analysis of interactions to determine underlying connections and insights, to help predict demand for services like hospitals enabling authorities to make better decisions about resource utilization, and to detect the changing patterns of customer behavior by analyzing data in near real-time, driving revenues and enhancing personalized experiences.

The AI of the future is expected to be more than just a tool that executes commands. It is expected to understand, reason, plan, and communicate in natural language. This is not a new idea – AI researchers have been pursuing this goal for decades, and the pursuit is more realistic now due to the recent breakthroughs in machine learning and neural networks.

The field of AI has been evolving rapidly, with breakthroughs in machine learning, neural networks, and deep learning. These technologies are being used to develop more advanced AI systems that can understand, learn, predict, adapt, and potentially operate autonomously. Systems that do visual applications can recognize faces in images and understand the content. Systems that understand speech and language can comprehend and respond to spoken language naturally.

However, the implementation of AI raises ethical issues. For example, AI systems can be biased if they are trained on data that is not representative of the broader population, or if the systems are designed in a way that reflects existing prejudices. The development of AI also raises concerns about job displacement, as AI systems can automate tasks previously done by humans.

In conclusion, AI is a rapidly evolving technology with the potential to revolutionize many aspects of our lives. Its development and implementation come with challenges and concerns, but its potential benefits are immense. AI is not just a tool for automating routine tasks, but a technology that can potentially understand, reason, and interact with the world in a human-like way. The future of AI is full of exciting possibilities and is a field that will continue to grow and develop in the coming years.


"""
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


print(generate_text("AI is", 20, model, max_sequence_len))
```
## The output:
```
Epoch 1/100
23/23 [==============================] - 4s 50ms/step - loss: 5.7769 - accuracy: 0.0385
Epoch 2/100
23/23 [==============================] - 2s 78ms/step - loss: 5.3949 - accuracy: 0.0454
Epoch 3/100
23/23 [==============================] - 3s 119ms/step - loss: 5.2394 - accuracy: 0.0440
Epoch 4/100
23/23 [==============================] - 5s 194ms/step - loss: 5.1695 - accuracy: 0.0605
Epoch 5/100
23/23 [==============================] - 3s 139ms/step - loss: 5.0897 - accuracy: 0.0688
Epoch 6/100
23/23 [==============================] - 3s 127ms/step - loss: 4.9687 - accuracy: 0.0757
Epoch 7/100
23/23 [==============================] - 3s 134ms/step - loss: 4.8711 - accuracy: 0.0853
Epoch 8/100
23/23 [==============================] - 3s 112ms/step - loss: 4.7306 - accuracy: 0.0963
Epoch 9/100
23/23 [==============================] - 3s 110ms/step - loss: 4.5740 - accuracy: 0.1169
Epoch 10/100
23/23 [==============================] - 3s 135ms/step - loss: 4.3777 - accuracy: 0.1403
Epoch 11/100
23/23 [==============================] - 3s 121ms/step - loss: 4.1792 - accuracy: 0.1582
Epoch 12/100
23/23 [==============================] - 3s 114ms/step - loss: 3.9618 - accuracy: 0.1939
Epoch 13/100
23/23 [==============================] - 3s 122ms/step - loss: 3.7579 - accuracy: 0.2077
Epoch 14/100
23/23 [==============================] - 3s 116ms/step - loss: 3.5484 - accuracy: 0.2435
Epoch 15/100
23/23 [==============================] - 3s 136ms/step - loss: 3.3647 - accuracy: 0.2682
Epoch 16/100
23/23 [==============================] - 3s 125ms/step - loss: 3.1696 - accuracy: 0.2957
Epoch 17/100
23/23 [==============================] - 2s 102ms/step - loss: 2.9735 - accuracy: 0.3191
Epoch 18/100
23/23 [==============================] - 3s 128ms/step - loss: 2.7924 - accuracy: 0.3549
Epoch 19/100
23/23 [==============================] - 4s 189ms/step - loss: 2.6219 - accuracy: 0.3989
Epoch 20/100
23/23 [==============================] - 3s 133ms/step - loss: 2.4657 - accuracy: 0.4264
Epoch 21/100
23/23 [==============================] - 3s 144ms/step - loss: 2.3076 - accuracy: 0.4787
Epoch 22/100
23/23 [==============================] - 3s 108ms/step - loss: 2.1626 - accuracy: 0.5172
Epoch 23/100
23/23 [==============================] - 2s 107ms/step - loss: 2.0145 - accuracy: 0.5626
Epoch 24/100
23/23 [==============================] - 3s 142ms/step - loss: 1.8769 - accuracy: 0.6424
Epoch 25/100
23/23 [==============================] - 3s 125ms/step - loss: 1.7488 - accuracy: 0.6836
Epoch 26/100
23/23 [==============================] - 3s 114ms/step - loss: 1.6228 - accuracy: 0.7235
Epoch 27/100
23/23 [==============================] - 2s 104ms/step - loss: 1.5061 - accuracy: 0.7469
Epoch 28/100
23/23 [==============================] - 2s 97ms/step - loss: 1.4067 - accuracy: 0.8006
Epoch 29/100
23/23 [==============================] - 3s 127ms/step - loss: 1.3038 - accuracy: 0.8006
Epoch 30/100
23/23 [==============================] - 3s 145ms/step - loss: 1.2061 - accuracy: 0.8377
Epoch 31/100
23/23 [==============================] - 3s 143ms/step - loss: 1.1271 - accuracy: 0.8487
Epoch 32/100
23/23 [==============================] - 3s 107ms/step - loss: 1.0433 - accuracy: 0.8803
Epoch 33/100
23/23 [==============================] - 2s 85ms/step - loss: 0.9687 - accuracy: 0.8927
Epoch 34/100
23/23 [==============================] - 2s 97ms/step - loss: 0.9037 - accuracy: 0.8996
Epoch 35/100
23/23 [==============================] - 2s 85ms/step - loss: 0.8381 - accuracy: 0.9065
Epoch 36/100
23/23 [==============================] - 2s 77ms/step - loss: 0.7803 - accuracy: 0.9147
Epoch 37/100
23/23 [==============================] - 2s 75ms/step - loss: 0.7337 - accuracy: 0.9188
Epoch 38/100
23/23 [==============================] - 2s 100ms/step - loss: 0.6855 - accuracy: 0.9230
Epoch 39/100
23/23 [==============================] - 3s 125ms/step - loss: 0.6386 - accuracy: 0.9298
Epoch 40/100
23/23 [==============================] - 4s 195ms/step - loss: 0.6021 - accuracy: 0.9395
Epoch 41/100
23/23 [==============================] - 4s 173ms/step - loss: 0.5659 - accuracy: 0.9422
Epoch 42/100
23/23 [==============================] - 4s 157ms/step - loss: 0.5340 - accuracy: 0.9381
Epoch 43/100
23/23 [==============================] - 3s 129ms/step - loss: 0.5059 - accuracy: 0.9477
Epoch 44/100
23/23 [==============================] - 3s 120ms/step - loss: 0.4779 - accuracy: 0.9519
Epoch 45/100
23/23 [==============================] - 3s 150ms/step - loss: 0.4507 - accuracy: 0.9422
Epoch 46/100
23/23 [==============================] - 3s 142ms/step - loss: 0.4283 - accuracy: 0.9532
Epoch 47/100
23/23 [==============================] - 4s 189ms/step - loss: 0.4053 - accuracy: 0.9532
Epoch 48/100
23/23 [==============================] - 4s 182ms/step - loss: 0.3849 - accuracy: 0.9574
Epoch 49/100
23/23 [==============================] - 3s 146ms/step - loss: 0.3663 - accuracy: 0.9560
Epoch 50/100
23/23 [==============================] - 3s 121ms/step - loss: 0.3487 - accuracy: 0.9574
Epoch 51/100
23/23 [==============================] - 3s 145ms/step - loss: 0.3334 - accuracy: 0.9601
Epoch 52/100
23/23 [==============================] - 3s 132ms/step - loss: 0.3185 - accuracy: 0.9629
Epoch 53/100
23/23 [==============================] - 4s 188ms/step - loss: 0.3069 - accuracy: 0.9587
Epoch 54/100
23/23 [==============================] - 3s 114ms/step - loss: 0.2964 - accuracy: 0.9560
Epoch 55/100
23/23 [==============================] - 2s 106ms/step - loss: 0.2828 - accuracy: 0.9587
Epoch 56/100
23/23 [==============================] - 3s 119ms/step - loss: 0.2725 - accuracy: 0.9601
Epoch 57/100
23/23 [==============================] - 3s 117ms/step - loss: 0.2622 - accuracy: 0.9574
Epoch 58/100
23/23 [==============================] - 3s 129ms/step - loss: 0.2537 - accuracy: 0.9629
Epoch 59/100
23/23 [==============================] - 3s 114ms/step - loss: 0.2470 - accuracy: 0.9601
Epoch 60/100
23/23 [==============================] - 3s 146ms/step - loss: 0.2390 - accuracy: 0.9601
Epoch 61/100
23/23 [==============================] - 3s 120ms/step - loss: 0.2316 - accuracy: 0.9601
Epoch 62/100
23/23 [==============================] - 2s 77ms/step - loss: 0.2230 - accuracy: 0.9642
Epoch 63/100
23/23 [==============================] - 3s 139ms/step - loss: 0.2168 - accuracy: 0.9656
Epoch 64/100
23/23 [==============================] - 3s 146ms/step - loss: 0.2096 - accuracy: 0.9629
Epoch 65/100
23/23 [==============================] - 3s 128ms/step - loss: 0.2042 - accuracy: 0.9656
Epoch 66/100
23/23 [==============================] - 3s 129ms/step - loss: 0.1987 - accuracy: 0.9587
Epoch 67/100
23/23 [==============================] - 2s 105ms/step - loss: 0.1941 - accuracy: 0.9629
Epoch 68/100
23/23 [==============================] - 2s 88ms/step - loss: 0.1891 - accuracy: 0.9656
Epoch 69/100
23/23 [==============================] - 3s 113ms/step - loss: 0.1855 - accuracy: 0.9642
Epoch 70/100
23/23 [==============================] - 3s 129ms/step - loss: 0.1816 - accuracy: 0.9642
Epoch 71/100
23/23 [==============================] - 3s 142ms/step - loss: 0.1766 - accuracy: 0.9642
Epoch 72/100
23/23 [==============================] - 3s 125ms/step - loss: 0.1743 - accuracy: 0.9587
Epoch 73/100
23/23 [==============================] - 3s 116ms/step - loss: 0.1687 - accuracy: 0.9642
Epoch 74/100
23/23 [==============================] - 3s 152ms/step - loss: 0.1659 - accuracy: 0.9629
Epoch 75/100
23/23 [==============================] - 2s 109ms/step - loss: 0.1628 - accuracy: 0.9629
Epoch 76/100
23/23 [==============================] - 2s 102ms/step - loss: 0.1589 - accuracy: 0.9629
Epoch 77/100
23/23 [==============================] - 2s 99ms/step - loss: 0.1565 - accuracy: 0.9656
Epoch 78/100
23/23 [==============================] - 2s 94ms/step - loss: 0.1539 - accuracy: 0.9629
Epoch 79/100
23/23 [==============================] - 2s 79ms/step - loss: 0.1506 - accuracy: 0.9642
Epoch 80/100
23/23 [==============================] - 2s 86ms/step - loss: 0.1487 - accuracy: 0.9629
Epoch 81/100
23/23 [==============================] - 2s 76ms/step - loss: 0.1461 - accuracy: 0.9656
Epoch 82/100
23/23 [==============================] - 2s 72ms/step - loss: 0.1430 - accuracy: 0.9642
Epoch 83/100
23/23 [==============================] - 2s 84ms/step - loss: 0.1409 - accuracy: 0.9615
Epoch 84/100
23/23 [==============================] - 3s 123ms/step - loss: 0.1392 - accuracy: 0.9670
Epoch 85/100
23/23 [==============================] - 3s 111ms/step - loss: 0.1364 - accuracy: 0.9656
Epoch 86/100
23/23 [==============================] - 1s 62ms/step - loss: 0.1361 - accuracy: 0.9601
Epoch 87/100
23/23 [==============================] - 1s 60ms/step - loss: 0.1333 - accuracy: 0.9629
Epoch 88/100
23/23 [==============================] - 2s 80ms/step - loss: 0.1305 - accuracy: 0.9642
Epoch 89/100
23/23 [==============================] - 2s 75ms/step - loss: 0.1279 - accuracy: 0.9670
Epoch 90/100
23/23 [==============================] - 2s 83ms/step - loss: 0.1266 - accuracy: 0.9656
Epoch 91/100
23/23 [==============================] - 2s 80ms/step - loss: 0.1254 - accuracy: 0.9642
Epoch 92/100
23/23 [==============================] - 2s 82ms/step - loss: 0.1240 - accuracy: 0.9684
Epoch 93/100
23/23 [==============================] - 2s 86ms/step - loss: 0.1214 - accuracy: 0.9670
Epoch 94/100
23/23 [==============================] - 2s 87ms/step - loss: 0.1213 - accuracy: 0.9629
Epoch 95/100
23/23 [==============================] - 2s 99ms/step - loss: 0.1202 - accuracy: 0.9615
Epoch 96/100
23/23 [==============================] - 2s 67ms/step - loss: 0.1180 - accuracy: 0.9642
Epoch 97/100
23/23 [==============================] - 2s 66ms/step - loss: 0.1175 - accuracy: 0.9629
Epoch 98/100
23/23 [==============================] - 2s 74ms/step - loss: 0.1158 - accuracy: 0.9642
Epoch 99/100
23/23 [==============================] - 2s 68ms/step - loss: 0.1153 - accuracy: 0.9642
Epoch 100/100
23/23 [==============================] - 1s 65ms/step - loss: 0.1132 - accuracy: 0.9656
AI is also used in the daily operations of many companies
```
