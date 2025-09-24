Core :

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Tokenization
word_vocab = sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab = sorted(set(tag for tags in target_texts for tag in tags))

word2idx = {word: i + 1 for i, word in enumerate(word_vocab)}  # Start from 1
tag2idx = {tag: i for i, tag in enumerate(tag_vocab)}

# Convert text to sequences
encoder_input_data = [[word2idx[word] for word in sent.split()] for sent in input_texts]
decoder_output_data = [[tag2idx[tag] for tag in tags] for tags in target_texts]

# Pad sequences
max_len = max(len(seq) for seq in encoder_input_data)
encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_len, padding='post')
decoder_output_data = pad_sequences(decoder_output_data, maxlen=max_len, padding='post')

# Reshape decoder output for sparse_categorical_crossentropy
decoder_output_data = np.expand_dims(decoder_output_data, -1)

# Model
vocab_size = len(word2idx) + 1
tag_size = len(tag2idx)

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embed = Embedding(input_dim=vocab_size, output_dim=32)(encoder_inputs)
encoder_lstm = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embed)

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embed = Embedding(input_dim=vocab_size, output_dim=32)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=[state_h, state_c])
decoder_dense = Dense(tag_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compile model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit([encoder_input_data, encoder_input_data], decoder_output_data, epochs=10, batch_size=2)

Output :
<img width="661" height="375" alt="image" src="https://github.com/user-attachments/assets/08a6b45b-676a-4150-ba43-c3a558dea13f" />
