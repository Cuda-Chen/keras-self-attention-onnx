import keras
from keras_self_attention import SeqSelfAttention
import keras2onnx

model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=10000,
                                 output_dim=300,
                                 mask_zero=True))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                       return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(keras.layers.Dense(units=5))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)
model.summary()

model.save('self_attention.h5')

onnx_model = keras2onnx.convert_keras(model, 'self_attention', debug_mode=1)
keras2onnx.save_model(onnx_model, 'self_attention.onnx')
