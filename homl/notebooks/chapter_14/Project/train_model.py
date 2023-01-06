import os
from contextlib import ExitStack

import tensorflow as tf
from tensorflow.keras import (
    Sequential,
    layers,
    optimizers,
    callbacks,
)
import tensorflow_hub as hub
from tensorflow.train import Example, Feature, Features, BytesList, Int64List

# Local Libraries
import ml_learning_rate
import ml_functions

BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 1000
DATASET_PATH = tf.io.gfile.join(os.environ["DATA_PATH"], "lung_colon_histopathology")
RAW_DATASET_PATH = tf.io.gfile.join(DATASET_PATH, "raw_data")
TFRECORDS_PATH = tf.io.gfile.join(DATASET_PATH, "tfrecord_data")
MODEL_PATH = os.path.join("..", "..", "models", "chapter_14")
SEED = 1992

# Determine the organ the sample is from
ORGAN_CLASSES = {
    0: "Colon",
    1: "Lung",
}

# Determine if the tissue is benign, adenocarcinoma or scamous cell carcinoma
TYPE_TISSUE = {
    0: "Benign",
    1: "Adenocarcinoma",
    2: "Squamous Cell Carcinoma",
}

IMG_SIZE = (768, 768)
IMG_CHANNELS = 3
NEW_IMG_SIZE = [448, 448, 3]

# Model
MODEL_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"


def get_key_from_value(dictionary, value):
    return [key for key, v in dictionary.items() if v == value][0]


def get_folders_images(filepath):
    name_folders = tf.io.gfile.listdir(filepath)
    path_folders = [tf.io.gfile.join(filepath, folder) for folder in name_folders]

    list_folders = {folder: path for folder, path in zip(name_folders, path_folders)}
    return list_folders


def create_example(image, organ_label, tissue_label):
    image_data = tf.io.serialize_tensor(image)
    feature = {
        "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),
        "organ_label": Feature(int64_list=Int64List(value=[organ_label.numpy()])),
        "tissue_label": Feature(int64_list=Int64List(value=[tissue_label.numpy()])),
    }

    return Example(features=Features(feature=feature))


def save_protobufs(dataset, type_set="train", n_shards=10):

    set_folder = tf.io.gfile.join(TFRECORDS_PATH, type_set)

    tf.io.gfile.makedirs(set_folder)
    file_paths = [tf.io.gfile.join(set_folder, filepath) for filepath in files]

    if type_set == "train":
        dataset.shuffle(SHUFFLE_BUFFER)
    files = [
        f"{type_set}.tfrecord-{shard.numpy() + 1:02d}-of-{n_shards:02d}"
        for shard in tf.range(n_shards)
    ]

    with ExitStack() as stack:
        writers = [
            stack.enter_context(tf.io.TFRecordWriter(file)) for file in file_paths
        ]
        for index, (image, organ_label, tissue_label) in dataset.enumerate():
            shard = index % n_shards
            example = create_example(image, organ_label, tissue_label)
            writers[shard].write(example.SerializeToString())
    return file_paths


def get_folders_tfrecords(type_set="train"):
    folder = tf.io.gfile.join(TFRECORDS_PATH, type_set)
    files = tf.io.gfile.listdir(folder)
    list_files = [tf.io.gfile.join(folder, filepath) for filepath in files]
    return list_files


def add_colon_label(image, label):
    return (image, 0, label)


def add_lung_label(image, label):
    return (image, 1, label)


def save_images_protobufs():

    # Verify if the folder for the TFRecords exist to process the data
    if not tf.io.gfile.exists(TFRECORDS_PATH):

        tf.io.gfile.makedirs(TFRECORDS_PATH)

        list_folders = get_folders_images(RAW_DATASET_PATH)

        # Create the dataset per folder, split the dataset into train, valid and test sets
        # Add the organ label
        # New dimension is (image, organ_label, tissue_label)
        # organ_label -> 0: Colon, 1: Lung
        # tissue_label -> 0: Benign, 1: Adenocarcinoma, 2: Squamous Cell Carcinoma
        colon_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            list_folders["colon_images"], batch_size=None, image_size=IMG_SIZE
        )
        train_colon, valid_colon, test_colon = ml_functions.balanced_split(
            colon_dataset
        )
        train_colon = train_colon.map(add_colon_label, num_parallel_calls=AUTOTUNE)
        valid_colon = valid_colon.map(add_colon_label, num_parallel_calls=AUTOTUNE)
        test_colon = test_colon.map(add_colon_label, num_parallel_calls=AUTOTUNE)

        lung_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            list_folders["lung_images"], batch_size=None, image_size=IMG_SIZE
        )
        train_lung, valid_lung, test_lung = ml_functions.balanced_split(lung_dataset)
        train_lung = train_lung.map(add_lung_label, num_parallel_calls=AUTOTUNE)
        valid_lung = valid_lung.map(add_lung_label, num_parallel_calls=AUTOTUNE)
        test_lung = test_lung.map(add_lung_label, num_parallel_calls=AUTOTUNE)

        # Combines the datasets into a single one
        train_set = train_colon.concatenate(train_lung)
        valid_set = valid_colon.concatenate(valid_lung)
        test_set = test_colon.concatenate(test_lung)

        train_paths = save_protobufs(train_set, "train")
        valid_paths = save_protobufs(valid_set, "valid")
        test_paths = save_protobufs(test_set, "test")
    else:
        train_paths = get_folders_tfrecords("train")
        valid_paths = get_folders_tfrecords("valid")
        test_paths = get_folders_tfrecords("test")

    return train_paths, valid_paths, test_paths


def get_record_multilabel(tfrecord):
    feature_descriptions = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "organ_label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "tissue_label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }

    example = tf.io.parse_single_example(tfrecord, feature_descriptions)
    image = tf.io.parse_tensor(example["image"], out_type=tf.float32)
    image = tf.reshape(image, shape=[IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS])
    image = tf.image.resize(image, size=[NEW_IMG_SIZE[0], NEW_IMG_SIZE[1]])
    return image, (example["organ_label"], example["tissue_label"])


def get_record(tfrecord):
    feature_descriptions = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "organ_label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "tissue_label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }

    example = tf.io.parse_single_example(tfrecord, feature_descriptions)
    image = tf.io.parse_tensor(example["image"], out_type=tf.float32)
    image = tf.reshape(image, shape=[IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS])
    image = tf.image.resize(image, size=[NEW_IMG_SIZE[0], NEW_IMG_SIZE[1]])
    return image, example["tissue_label"]


def get_dataset(file_paths, cache=True, shuffle_buffer=None, multi_label=False):
    dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=AUTOTUNE)

    if cache:
        dataset = dataset.cache()
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)

    if multi_label:
        dataset = (
            dataset.map(get_record_multilabel, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
    else:
        dataset = (
            dataset.map(get_record, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )

    return dataset


def train_model(
    train_set,
    valid_set,
    test_set,
    epochs,
    learning_rate,
    trainable=False,
    lr_function=None,
):
    # Normalization of the data
    def normalize(image, label):
        norm_image = image / 255.0
        return (norm_image, label)

    train_set_normalized = train_set.map(normalize, num_parallel_calls=AUTOTUNE)
    valid_set_normalized = valid_set.map(normalize, num_parallel_calls=AUTOTUNE)
    test_set_normalized = test_set.map(normalize, num_parallel_calls=AUTOTUNE)

    # Load the ResNet model
    base_model = hub.KerasLayer(
        MODEL_URL,
        trainable=trainable,
    )

    model = Sequential(
        [layers.MaxPool2D(), base_model, layers.Dense(3, activation="softmax")]
    )

    # Compilation of the model
    optimizer_ = optimizers.Adam(learning_rate=learning_rate)
    metrics_ = ["accuracy"]
    loss_ = "sparse_categorical_crossentropy"

    model.compile(optimizer=optimizer_, metrics=metrics_, loss=loss_)

    # Callbacks
    exponential_decay_fn = ml_learning_rate.exponential_decay_with_warmup(
        lr_start=learning_rate,
        lr_max=learning_rate * 10,
        lr_min=learning_rate / 10,
    )
    lr_scheduler_cb = callbacks.LearningRateScheduler(exponential_decay_fn)

    folder_logs = tf.io.gfile.join(
        "..", "..", "..", "reports", "logs", "chapter_14", "lung_colon_histopathology"
    )
    logdir = ml_functions.get_logdir(path_folder=folder_logs)
    tensorboard_cb = callbacks.TensorBoard(log_dir=logdir)

    model_path = tf.io.gfile.join(MODEL_PATH, "lung_colon_histopathology.h5")
    model_checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=model_path, save_best_only=True
    )

    early_stopping_cb = callbacks.EarlyStopping(patience=5)

    callbacks_list = [
        lr_scheduler_cb,
        tensorboard_cb,
        model_checkpoint_cb,
        early_stopping_cb,
    ]

    # Train the model
    model.fit(
        train_set_normalized,
        validation_data=valid_set_normalized,
        epochs=epochs,
        callbacks=callbacks_list,
    )

    evaluation = model.evaluate(test_set_normalized, verbose=0)

    return model, evaluation


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    train_files, valid_files, test_files = save_images_protobufs()
    train_set = get_dataset(train_files, shuffle_buffer=SHUFFLE_BUFFER)
    valid_set = get_dataset(valid_files)
    test_set = get_dataset(test_files)

    model, evaluation = train_model(
        train_set,
        valid_set,
        test_set,
        epochs=5,
        learning_rate=1e-4,
        trainable=False,
        lr_function=None,
    )

    print(evaluation)
