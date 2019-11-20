import keras 
from keras import models
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

n_samples, n_features = X.shape
n_samples, n_targets = Y.shape

# --- neural network configuration
model = models.Sequential()
model.add(Dense(128, activation='relu', input_shape=(n_features,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_targets, activation='softmax'))

# --- load model from previous training as initial weights : TEST PURPOSE
#model.load_weights('model_ap_master.hd5')

callbacks_list = [ 
    keras.callbacks.EarlyStopping( 
        monitor='val_accuracy', 
        patience = 10, 
    ),
    keras.callbacks.ModelCheckpoint( 
        filepath='model_ap_master.hd5', 
        monitor='val_accuracy',
        save_best_only=True 
    ),
#     keras.callbacks.TensorBoard(
#         log_dir='TensorBoard', 
#         histogram_freq=1, 
#         #embeddings_freq=1, 
#     )    
]

# --- training
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    X, Y, 
    epochs = 100, 
    batch_size=3000, 
    validation_split=0.20,
    callbacks = callbacks_list,
    verbose = False
)

# from keras.utils import plot_model
# plot_model(model, show_shapes=True, to_file='model_ap_master.png')

# --- plot 
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.figure(figsize = (13,5))
plt.subplot(1,2,1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, acc, 'r', label='accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Accuracy and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
plt.close()
