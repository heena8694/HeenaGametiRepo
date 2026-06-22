import numpy as np
import string

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

text = text.lower()

for p in string.punctuation:
    text = text.replace(p, "")

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
print("vocabulary size", total_words)

input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding='pre'
)

x = input_sequences[:, :-1]
y = input_sequences[:, :-1]

y = to_categorical(y, num_classes=total_words)

x_train,x_val,y_train,y_val = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = sequential()

model.add(
    Embedding(
        total_words,
        100,
        input_length=max_sequence_len - 1
    )
)

model.add(
    Dense(
        total_words,
        activation='softmax'
    )
)

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)
model.summary()

early_stop = EarlyStopping(
    monitor='vals_loss',
    patience=3,
    restore_best_weights=True
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_val,y_val),
    epochs=20,
    batch_size=128,
    callbacks=[early_stop]
)

def genrate_text(seed_text, next_words):

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences(
            [seed_text]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )
        predicted = np.argmax(
            model.predict(token_list, verbose=0),
            axis=-1
        )[0]

        output_word = " "

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text
print("\nGenrated Text:\n")
print(genrate_text("to be", 20))
print()
print(genrate_text("love is", 20))
print()
print(genrate_text("the king", 20))
      
