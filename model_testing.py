import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import visualkeras
from pathlib import Path

test_dir = './data/test_eyes'
batch_size = 16
img_height = 180
img_width = 180

# Dataset de imagenes de prueba
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    seed=100,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# Array de labels
Y_test = np.array([])
for image_batch, labels_batch in test_ds:
    labels = np.array(labels_batch)
    Y_test = np.append(Y_test, labels)


Y_test = Y_test.astype(int)
print(Y_test)

model = tf.keras.models.load_model('modelo.h5', custom_objects={
                                   'Functional': tf.keras.models.Model})

predictions = model.predict(test_ds)
Y_pred = np.argmax(predictions, axis=-1)
print(Y_pred)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")


cm = confusion_matrix(y_true=Y_test, y_pred=Y_pred)


cm_plot_labels = ["Closed", "Open"]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")

print(classification_report(Y_test, y_pred=Y_pred))

# Grafico detallado de la arquitectura del modelo
tf.keras.utils.plot_model(model, to_file="modelo.png",
                          show_shapes=True, show_layer_names=True)

# Grafico basico de la arquitectura del modelo
visualkeras.layered_view(
    model, legend=True, draw_volume=True, to_file='arquitecture.png',  scale_xy=3, scale_z=1, max_z=1000)


Path("modelo.png").rename("images/modelo.png")
Path("arquitecture.png").rename("images/arquitecture.png")
Path("confusion_matrix.png").rename("images/confusion_matrix.png")
