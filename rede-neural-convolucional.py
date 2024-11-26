import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import os

# Caminhos das pastas de treino e validação
train_dir = "C:/Users/mathe/source/Projetos com OpenCV/Reconhecimento de libras/archive/asl_alphabet_train/asl_alphabet_train"
test_dir = "C:/Users/mathe/source/Projetos com OpenCV/Reconhecimento de libras/archive/asl_alphabet_test/asl_alphabet_test"

train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Implementação do modelo de Rede Neural Convolucional
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(29, activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1
)

model.save("modelo_libras.h5")
