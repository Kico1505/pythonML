import threading
import cv2 as cv
import numpy as np
import matplotlib.pyplot as pymat
from keras import *
import os
from tkinter import *
from tkinter import filedialog, ttk
import random
import time

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images = training_images / 255
testing_images = testing_images / 255

epochs = 10

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

model = models.load_model("image_classifier.keras")

global progress
global bar


def predict_single_image():
    image = cv.imread(filedialog.askopenfilename())
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    pymat.subplot(1, 1, 1)
    pymat.xticks([])
    pymat.yticks([])
    pymat.imshow(image, cmap=pymat.cm.binary)

    prediction = model.predict(np.array([image]) / 255)
    index = np.argmax(prediction)

    pymat.xlabel(class_names[index] + " " + str(round(prediction[0][index] * 100, 2)) + "%")

    pymat.show()


def set_epoch():
    global epochs
    global epoch_entry
    try:
        val = int(epoch_entry.get())
        if 0 < val <= 500:
            epochs = val
            epoch_label.config(text="Epoch is now: " + str(epochs))
        else:
            epoch_label.config(text="Value out of range(1-500)")
    except Exception:
        epoch_label.config(text="Invalid Input")

# This method will generate a new model and save it, the model is already generated and is slow to generate so
# it is not recommended to rerun it
def generate_model_vizualization():
    window = threading.Thread(target=generate_data)
    window.start()

    modelx = models.Sequential()
    modelx.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    modelx.add(layers.MaxPooling2D(2, 2))
    modelx.add(layers.Conv2D(64, (3, 3), activation="relu"))
    modelx.add(layers.MaxPooling2D(2, 2))
    modelx.add(layers.Conv2D(64, (3, 3), activation="relu"))
    modelx.add(layers.Flatten())
    modelx.add(layers.Dense(64, activation="relu"))
    modelx.add(layers.Dense(10, activation="softmax"))

    base_learning_rate = 0.0001
    modelx.compile(optimizer=optimizers.Adam(learning_rate=base_learning_rate),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

    history = modelx.fit(training_images, training_labels, epochs=epochs,
                         validation_data=(testing_images, testing_labels))

    bar["value"] = float(99.9)
    progress.destroy()
    window.join()

    accuracy = history.history["accuracy"]
    validation_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    validation_loss = history.history["val_loss"]
    epochs_range = range(epochs)

    pymat.figure(figsize=(15, 15))
    pymat.subplot(2, 2, 1)
    pymat.plot(epochs_range, accuracy, label="Training Accuracy")
    pymat.plot(epochs_range, validation_accuracy, label="Validation Accuracy")
    pymat.legend(loc="lower right")
    pymat.title("Training and Validation Accuracy")

    pymat.subplot(2, 2, 2)
    pymat.plot(epochs_range, loss, label="Training Loss")
    pymat.plot(epochs_range, validation_loss, label="Validation Loss")
    pymat.legend(loc="upper right")
    pymat.title("Training and Validation Loss")

    # modelx.save("image_classifier.keras")

    pymat.show()

def generate_new_model():
    window = threading.Thread(target=generate_data)
    window.start()

    modelx = models.Sequential()
    modelx.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    modelx.add(layers.MaxPooling2D(2, 2))
    modelx.add(layers.Conv2D(64, (3, 3), activation="relu"))
    modelx.add(layers.MaxPooling2D(2, 2))
    modelx.add(layers.Conv2D(64, (3, 3), activation="relu"))
    modelx.add(layers.Flatten())
    modelx.add(layers.Dense(64, activation="relu"))
    modelx.add(layers.Dense(10, activation="softmax"))

    base_learning_rate = 0.0001
    modelx.compile(optimizer=optimizers.Adam(learning_rate=base_learning_rate),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

    modelx.fit(training_images, training_labels, epochs=epochs,
                         validation_data=(testing_images, testing_labels))

    bar["value"] = float(99.9)
    window.join()
    progress.destroy()

    folder_dir = "./images"
    x = 0

    for images in os.listdir(folder_dir):
        image = cv.imread(folder_dir + "/" + images)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        pymat.subplot(4, 6, x + 1)
        pymat.xticks([])
        pymat.yticks([])
        pymat.imshow(image, cmap=pymat.cm.binary)

        prediction = modelx.predict(np.array([image]) / 255)

        index = np.argmax(prediction)
        pymat.xlabel(class_names[index] + " " + str(round(prediction[0][index] * 100, 2)) + "%")
        x += 1

    pymat.show()


# method that shows a random set of 25 images from the training data
def visualize_training_images():
    pick = random.choice(range(25, 6000, 25))
    for x in range(25):
        pymat.subplot(5, 5, x+1)
        pymat.xticks([])
        pymat.yticks([])
        pymat.imshow(training_images[pick + x], cmap=pymat.cm.binary)
        pymat.xlabel(class_names[training_labels[pick + x][0]])
    pymat.show()


def generate_predictions():
    folder_dir = "./images"
    x = 0

    for images in os.listdir(folder_dir):
        image = cv.imread(folder_dir + "/" + images)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        pymat.subplot(4, 6, x + 1)
        pymat.xticks([])
        pymat.yticks([])
        pymat.imshow(image, cmap=pymat.cm.binary)

        prediction = model.predict(np.array([image]) / 255)

        index = np.argmax(prediction)
        pymat.xlabel(class_names[index] + " " + str(round(prediction[0][index] * 100, 2)) + "%")
        x += 1

    pymat.show()


def generate_data():
    global progress
    global bar
    progress = Toplevel()
    progress.title("Generating New Data")
    progress.geometry("400x100")
    progress_label = Label(progress, text="Generating Data, Please Wait...\n"
                                         "Average Expected Wait Time: " + str(epochs / 10) + " Minutes")
    progress_label.place(x=100, y=10)
    bar = ttk.Progressbar(progress)
    bar.place(x=100, y=60, width=200)
    root.update()
    finished = 0;
    while finished < epochs * 0.9:
        bar["value"] = (round(float(finished / epochs)*100, 2))
        finished += 1
        progress.update()
        time.sleep(5)


root = Tk()
root.title("Image Classification Application")
root.geometry("600x390")
root.resizable(False, False)

epoch_label = Label(root, text="Set Epoch (Training Batches)")
epoch_label.place(x=360, y=30)
epoch_entry = Entry(root, width=10)
epoch_entry.place(x=398, y=60)
epoch_set = Button(root, text="Set", height=1,width=10, command=set_epoch)
epoch_set.place(x=390, y=90)

single_predict = Button(root, text="Select File for Prediction\n                        ",
                       height=5, width=30, command=predict_single_image)
single_predict.place(x=50, y=30)

group_predict = Button(root, text="Predict Pre-Made Group of Images\n                   ",
                      height=5, width=30, command=generate_predictions)
group_predict.place(x=50, y=150)

training_data = Button(root, text="Visualize a Group of Training Data\n          ",
                      height=5, width=30, command=visualize_training_images)
training_data.place(x=325, y=150)

model_visual = Button(root, text="Visualize Model Accuracy and Loss\n(Takes a While)    ",
                      height=5, width=30, command=
                      threading.Thread(target=generate_model_vizualization, daemon=True).start)
model_visual.place(x=50, y=270)

model_visual = Button(root, text="Create a New Model and\nPredict Images\n(Takes a While)",
                      height=5, width=30, command=
                      threading.Thread(target=generate_new_model, daemon=True).start)
model_visual.place(x=325, y=270)

root.mainloop()
