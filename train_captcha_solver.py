from itertools import repeat
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from keras import callbacks, layers
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from tensorflow import random

# Set random seeds. Remove if non-deterministic required.
RANDOM_SEED = 42  # set to None if non-deterministic
SHUFFLE = False  # set to true if non-deterministic


class SaveBestModel(callbacks.Callback):
    def __init__(self, path_fmt_save_best):
        super(SaveBestModel, self).__init__()
        self.path_fmt_save_best = path_fmt_save_best
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs["val_loss"]
        if val_loss < self.best_val_loss:
            # Delete previous best model.
            if self.best_epoch > 0:
                try:
                    Path(self.path_fmt_save_best.format(epoch=self.best_epoch, val_loss=self.best_val_loss)).unlink()
                except FileNotFoundError:
                    pass
            self.best_epoch = epoch + 1
            self.best_val_loss = val_loss
            self.model.save(
                self.path_fmt_save_best.format(
                    epoch=epoch + 1,
                    val_loss=val_loss,
                )
            )


def loadCaptchasForTraining(paths_captcha_imgs, shape_captcha_imgs, captcha_length, charset):

    # Init array to store CAPTCHA imgs (num_imgs, height, width, channels)
    captcha_imgs = np.zeros(
        (
            len(paths_captcha_imgs),
            *shape_captcha_imgs,  # in order: height, width, channels
        ),
        dtype=np.float64,
    )

    # Init array to store CAPTCHA text labels (num_imgs, captcha_len, charset_len)
    captcha_labels = np.zeros(
        (
            len(paths_captcha_imgs),
            captcha_length,
            len(charset),
        ),
        dtype=np.uint8,
    )

    for i, path_img in enumerate(paths_captcha_imgs):

        if len(path_img.stem) == captcha_length:

            # Load CAPTCHA img in grayscale (they have ndims=2 in OpenCV), normalize to range 0-1, resize to (50, 200), and
            # add channel dimension.
            img = cv2.imread(str(path_img), cv2.IMREAD_GRAYSCALE).astype(np.float64)
            img /= 255.0
            img = cv2.resize(img, (shape_captcha_imgs[1], shape_captcha_imgs[0]), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, axis=-1)

            # Grab the image name (contains the characters in the partiular CAPTCHA).
            text = path_img.stem

            # Initialize a one-hot label array for the CAPTCHA's label, initially all zeros.
            label = np.zeros((captcha_length, len(charset)), dtype=np.uint8)

            # Go over the image name characters and set the corresponding label indices to one.
            for j, char in enumerate(text):
                index = charset.index(char)  # get the index in the character set
                label[j, index] = 1  # set the corresponding index to one

            # Update the CAPTCHA img and label arrays.
            captcha_imgs[i, ...] = img
            captcha_labels[i, ...] = label

    return captcha_imgs, captcha_labels


def createCaptchaSolver(img_shape, captcha_length, charset):
    # Create the image input layer
    img_input = layers.Input(shape=img_shape, name="ImageInput")

    # Setup the first convolutional layer
    l1b1_conv2d = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="Conv2D_11")(img_input)
    l1b2_maxpool = layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool_12")(l1b1_conv2d)
    l1b3_batchnorm = layers.BatchNormalization(name="BatchNorm_13")(l1b2_maxpool)

    # Setup the second convolutional layer
    l2b1_conv2d = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="Conv2D_21")(l1b3_batchnorm)
    l2b2_maxpool = layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool_22")(l2b1_conv2d)
    l2b3_batchnorm = layers.BatchNormalization(name="BatchNorm_23")(l2b2_maxpool)

    # Setup the third convolutional layer
    l3b1_conv2d = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="Conv2D_31")(l2b3_batchnorm)
    l3b2_maxpool = layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool_32")(l3b1_conv2d)
    l3b3_batchnorm = layers.BatchNormalization(name="BatchNorm_33")(l3b2_maxpool)

    # Flatten the output of the third convolutional layer
    l4b1_flatten = layers.Flatten(name="Flatten_41")(l3b3_batchnorm)

    # Now we want dense layers. We want as many of these layers as the length of the CAPTCHA, and each of these layers
    # must have as many neurons as the characters in the character set.
    dense_outputs = []
    for i in range(captcha_length):
        l4b2_dense = layers.Dense(64, activation="relu", name=f"Dense_42_{i}")(l4b1_flatten)
        l4b3_batchnorm = layers.BatchNormalization(name=f"BatchNorm_43_{i}")(l4b2_dense)
        l5b1_dense = layers.Dense(len(charset), activation="softmax", name=f"Dense_44_{i}")(l4b3_batchnorm)
        dense_outputs.append(l5b1_dense)

    # Compile the model.
    model = Model(inputs=img_input, outputs=dense_outputs, name="CaptchaSolver")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    np.random.seed(RANDOM_SEED)
    random.set_seed(RANDOM_SEED)

    # All lowercase characters + digits that are used in the CAPTCHA images, number of characters in the CAPTCHA, and
    # the height, weight, and channels of the CAPTCHA images.
    CAPTCHA_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"
    CAPTCHA_LENGTH = 5
    CAPTCHA_SHAPE = (50, 200, 1)
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    SPLIT_PROPORTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}
    DIR_CAPTCHAS = Path("captcha-dataset/samples")
    DIR_RESULTS = Path("models")
    PATH_FMT_SAVE_BEST = str(DIR_RESULTS / "BEST-CaptchaSolver-epoch_{epoch:02d}-vloss_{val_loss:.2f}.h5")
    PATH_FMT_SAVE_LAST = str(DIR_RESULTS / "LAST-CaptchaSolver-epoch_{epoch:02d}-vloss_{val_loss:.2f}.h5")

    if not DIR_CAPTCHAS.exists():
        raise FileNotFoundError(f"Directory containing CAPTCHA images '{DIR_CAPTCHAS}' does not exist.")

    if not DIR_RESULTS.exists():
        DIR_RESULTS.mkdir()

    # Total CAPTCHA samples
    total_captchas = len(list(DIR_CAPTCHAS.glob("*.png")))

    if total_captchas == 0:
        raise FileNotFoundError(f"No CAPTCHA images found in '{DIR_CAPTCHAS}'.")

    #

    print(f"Total CAPTCHAs: {total_captchas}\nCharacter Set of {len(CAPTCHA_CHARSET)} = '{CAPTCHA_CHARSET}'")

    # Load CAPTCHAs as numpy arrays of images and labels
    captcha_imgs, captcha_labels = loadCaptchasForTraining(
        list(DIR_CAPTCHAS.glob("*.png")), CAPTCHA_SHAPE, CAPTCHA_LENGTH, CAPTCHA_CHARSET
    )

    # Create the model to solve CAPTCHAs
    model = createCaptchaSolver(CAPTCHA_SHAPE, CAPTCHA_LENGTH, CAPTCHA_CHARSET)

    # Split the data into tra/test/val. First we do the train-test split, and then we split the train set into
    # train/val.
    captcha_imgs_train, captcha_imgs_test, captcha_labels_train, captcha_labels_test = train_test_split(
        captcha_imgs, captcha_labels, test_size=SPLIT_PROPORTIONS["test"], random_state=RANDOM_SEED, shuffle=SHUFFLE
    )

    # At this point, the train test also has the validation set, but the train set is a subset of the full datset.
    # Therefore, the validation split which we defined in terms of the full dataset needs to be updated so that the
    # sampled data from the train set equals the validation proportion of the full dataset. That can be done by dividing
    # the full dataset proportion of validation split by the same of the train split, resulting in the full-dataset
    # equivalent proportion for validation data from the subset of train data.
    captcha_imgs_train, captcha_imgs_val, captcha_labels_train, captcha_labels_val = train_test_split(
        captcha_imgs_train,
        captcha_labels_train,
        test_size=SPLIT_PROPORTIONS["val"] / SPLIT_PROPORTIONS["train"],
        random_state=RANDOM_SEED,
        shuffle=SHUFFLE,
    )

    # Setup callbacks.
    saveBestModel = SaveBestModel(PATH_FMT_SAVE_BEST)
    earlyStopping = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=50,
        verbose=0,
        restore_best_weights=True,
    )

    # print(captcha_labels_train[:, 0, ...].shape)

    # Train the model.
    train_history = model.fit(
        captcha_imgs_train,
        [captcha_labels_train[:, i, :] for i in range(CAPTCHA_LENGTH)],
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(captcha_imgs_val, [captcha_labels_val[:, i, :] for i in range(CAPTCHA_LENGTH)]),
        callbacks=[saveBestModel, earlyStopping],
        verbose=1,
    )

    # Create a dataframe for the training history.
    train_history_df = pd.DataFrame(train_history.history)
    train_history_df.insert(0, "epoch", [x + 1 for x in train_history.epoch])

    # Get all column names ending with "_accuracy" and calculate the mean of those columns to get the average accuracy
    # across all CAPTCHA characters. Similarly get all those columns that start with val_ and end with _accuracy.
    accuracy_cols = [col for col in train_history_df.columns if col.endswith("_accuracy")]
    val_accuracy_cols = [
        col for col in train_history_df.columns if col.startswith("val_") and col.endswith("_accuracy")
    ]
    train_history_df.insert(2, "accuracy", train_history_df[accuracy_cols].mean(axis=1))
    train_history_df.insert(3, "val_accuracy", train_history_df[val_accuracy_cols].mean(axis=1))

    # Save the latest model.
    model.save(
        PATH_FMT_SAVE_LAST.format(
            epoch=train_history_df["epoch"].iloc[-1],
            val_loss=train_history_df["val_loss"].iloc[-1],
        )
    )

    # Plot the training history for accuracy.
    plt.figure()
    plt.plot(train_history_df["accuracy"], label="train_acc", color="b")
    plt.plot(train_history_df["val_accuracy"], label="val_acc", color="r")
    plt.gca().xaxis.get_major_locator().set_params(integer=True)  # ensure x-axis ticks are integers
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(DIR_RESULTS / "training_history_acc.png")

    # Plot the training history for loss on a new figure.
    plt.figure()
    plt.plot(train_history_df["loss"], label="train_loss", color="b")
    plt.plot(train_history_df["val_loss"], label="val_loss", color="r")
    plt.gca().xaxis.get_major_locator().set_params(integer=True)  # ensure x-axis ticks are integers
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(DIR_RESULTS / "training_history_loss.png")

    # Save the training history as an excel sheet.
    train_history_df.to_excel(DIR_RESULTS / "training_history.xlsx", index=False)

    print("\nView train session results in directory:", DIR_RESULTS.as_posix())

    print("\nEvaluating on test set...\n")

    # Evaluate on test set.
    test_metrics = model.evaluate(
        captcha_imgs_test,
        [captcha_labels_test[:, i, :] for i in range(CAPTCHA_LENGTH)],
        batch_size=BATCH_SIZE,
    )

    # Last 5 entries in test_metrics are the accuracies against each character. Average them out insert into the
    # test_metrics list.
    test_mean_accuracy = np.mean(test_metrics[-CAPTCHA_LENGTH:])
    test_metrics.insert(1, test_mean_accuracy)

    # Dump test metrics to YAML file.
    with open(DIR_RESULTS / "test_metrics.yaml", "w") as f:
        yaml.safe_dump(
            {
                "test_loss": float(test_metrics[0]),
                "test_accuracy": float(test_metrics[1]),
            },
            f,
            sort_keys=False,
        )

    print("\nTest metrics saved to YAML:", (DIR_RESULTS / "test_metrics.yaml").as_posix())


if __name__ == "__main__":
    main()
