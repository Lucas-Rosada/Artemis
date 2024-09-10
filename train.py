import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Diretórios de treinamento e validação
train_dir = 'C:/Users/Lucas/Desktop/sinais/DATABASE/training'
val_dir = 'C:/Users/Lucas/Desktop/sinais/DATABASE/val'

# Definindo parâmetros
img_height, img_width = 224, 224
batch_size = 32

# Gerador de dados para treinamento com aumento de dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Gerador de dados para validação
val_datagen = ImageDataGenerator(rescale=1./255)

# Carregar dados de treinamento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'  # Use 'categorical' se você tiver um one-hot encoding
)

# Carregar dados de validação
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'  # Use 'categorical' se você tiver um one-hot encoding
)

# Função para criar o modelo
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(train_generator.class_indices), activation='softmax'))  # Número de classes

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Criação do modelo
model = create_model()

# Definindo o número de épocas
epochs = 10

# Definindo o callback de EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Treinamento do modelo com EarlyStopping
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Salvar o modelo treinado
model.save('C:/Users/Lucas/Desktop/sinais/keras_model.h5')

print("Modelo treinado e salvo com sucesso!")