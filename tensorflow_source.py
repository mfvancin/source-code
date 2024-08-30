# Basic Concepts
#Tensors: The basic unit of data in TensorFlow. Tensors are multidimensional arrays that flow through the graph.
#Graph: A TensorFlow program is represented as a dataflow graph. Nodes in the graph represent operations (like addition or multiplication), and edges represent the tensors that flow between them.
#Session: A session encapsulates the control and state of the TensorFlow runtime.

# Creating Tensors
import tensorflow as tf

# Create a scalar tensor
scalar = tf.constant(7)

# Create a 1D tensor (vector)
vector = tf.constant([1, 2, 3, 4])

# Create a 2D tensor (matrix)
matrix = tf.constant([[1, 2, 3], [4, 5, 6]])

# Create a 3D tensor
tensor3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:", matrix)
print("3D Tensor:", tensor3d)

# Tensor Operations
# Tensor addition
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
sum_tensor = tf.add(a, b)

# Tensor multiplication
product_tensor = tf.matmul(a, b)

print("Sum Tensor:\n", sum_tensor.numpy())
print("Product Tensor:\n", product_tensor.numpy())

# Building a Simple Neural Network
from tensorflow.python.keras import datasets
mnist = datasets.mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten

# Create a Sequential model
model = Sequential()

# Flatten the input data
model.add(Flatten(input_shape=(28, 28)))

# Add hidden layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Add output layer
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

# Using Callbacks
from tensorflow.python.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Saving and Loading Models
model.save('my_model.h5')

from tensorflow.python.keras.models import load_model

loaded_model = load_model('my_model.h5')

# Using Pre-trained Models
from tensorflow.python.keras import applications
VGG16 = applications.VGG16

# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False)

# Add new layers on top for your specific task
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Create a new model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Custom Layers
import tensorflow as tf

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Example usage
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    CustomDense(128),
    tf.keras.layers.ReLU(),
    CustomDense(10),
])

# Custom Models
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Example usage
model = MyModel()

# Custom Training Loops
# Prepare dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Create model
model = MyModel()

# Compile model
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Custom training loop
for epoch in range(5):
    for step in range(len(x_train) // 32):
        x_batch = x_train[step * 32: (step + 1) * 32]
        y_batch = y_train[step * 32: (step + 1) * 32]

        with tf.GradientTape() as tape:
            predictions = model(x_batch)
            loss = loss_fn(y_batch, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f"Epoch {epoch + 1}: Loss: {loss.numpy()}")

# Callbacks
# ModelCheckpoint: Saves the model at regular intervals or when it improves.
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)

# EarlyStopping: Stops training when a monitored metric has stopped improving.
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# TensorBoard: Logs metrics for visualization in TensorBoard.
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./logs')

# Distributed Training
# MirroredStrategy: Copies all variables to each processor. Each processor computes gradients based on its subset of data.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = MyModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# MultiWorkerMirroredStrategy: Similar to MirroredStrategy, but allows training on multiple machines.
# TPUStrategy: For training on TPUs (Tensor Processing Units).
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = MyModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_dataset, epochs=5)

# Hyperparameter Tuning
# Using Keras Tuner:
from kerastuner import HyperModel

class MyHyperModel(HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

from kerastuner.tuners import RandomSearch

tuner = RandomSearch(
    MyHyperModel(),
    objective='val_accuracy',
    max_trials=5,
    directory='my_dir',
    project_name='helloworld'
)

tuner.search(x_train, y_train, epochs=5, validation_split=0.2)

best_model = tuner.get_best_models(num_models=1)[0]

# Setting up TensorFlow Serving:
model.save('saved_model/my_model')

# Serve the model using Docker
#docker run -p 8501:8501 --name=tf_model_serving --mount type=bind,source=$(pwd)/saved_model/my_model,target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving

# Make predictions
import requests
import json

data = json.dumps({"signature_name": "serving_default", "instances": x_test[:1].tolist()})
headers = {"content-type": "application/json"}
response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data, headers=headers)
print(response.json())

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Run inference using TensorFlow Lite
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data
input_data = np.array(x_test[0], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

# Working with Image Data
from tensorflow.python.keras import preprocessing
ImageDataGenerator = preprocessing.image.ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load data
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Fit the model
model.fit(train_generator, epochs=5)

# Regularization Techniques
# Dropout: Randomly drops a fraction of neurons during training.
model.add(tf.keras.layers.Dropout(0.5))

# L1 and L2 Regularization: Adds penalties for large weights.
from tensorflow.keras import regularizers

model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

# Batch Normalization: Normalizes activations in the network.
model.add(tf.keras.layers.BatchNormalization())

# Advanced Optimizers
# Adam: Combines the advantages of two other extensions of stochastic gradient descent.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# RMSprop: Handles the diminishing learning rates in Adagrad.
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# Learning Rate Schedules: Adjusts the learning rate during training.
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
model.fit(x_train, y_train, epochs=20, callbacks=[lr_schedule])
