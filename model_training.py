import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from pathlib import Path

data_dir = './data/train_eyes'
batch_size = 16
img_height = 180
img_width = 180

# Dataset de imagenes para el entrenamiento
# train_ds = [[16 imagenes con labels],[16 imagenes con labels]...]
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    validation_split=0.2,
    subset="training",
    seed=100,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# Dataset de imagenes para la validacion
# val_ds = [[16 imagenes con labels],[16 imagenes con labels]...]
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    validation_split=0.2,
    subset="validation",
    seed=100,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# Visualizar algunas imagenes del dataset de entrenamiento
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    print(len(images))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(32, (3, 3), padding='same', activation=layers.LeakyReLU()),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), padding='same', activation=layers.LeakyReLU()),
    layers.MaxPooling2D(),

    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(units=100, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# Entrenamiento
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    batch_size=16
)

# Recoleccion de resultados
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)


# Grafico de loss
sns.set_theme(palette="ch:s=.25,rot=-.25")
fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(data=loss, ax=ax, color="b", label='Training Loss')
sns.lineplot(data=val_loss, ax=ax, color="r", label='Validation Loss')
ax.set_xlabel("Épocas")
ax.set_ylabel("Loss")
plt.savefig("lineplot.png")
Path("lineplot.png").rename("images/lineplot.png")


# Guarda el modelo en archivo h5
model.save('modelo.h5')
