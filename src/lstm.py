#!/usr/bin/env python

"""
Train an LSTM or LLM model to predict the category that reviews belong to.
By using different command line arguments, the models and settings can be
adjusted. Part of this code was provided for this experiment, and thus we do not
claim full ownership of this code.
"""

import json
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Reshape, Bidirectional
from keras.initializers import Constant
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from util import parse_config, load_data, OffensiveWordReplaceOption


def create_default_config(create_file: bool = False) -> dict[str, str]:
    """Creates a config file with default parameters, and returns them"""
    default_config = {
        'data_dir': './data',
        'preprocessed': True,  # True, False
        'replace_option': 'none',  # none, replace, remove
        'embedding_file': './data/glove.6B.50d.json',
        'evaluation_set_while_training': 'dev',  # dev, test
        'seed': None,  # int or None
        'learning_rate': 0.00005,
        'batch_size': 16,
        'dropout': 0.1,
        'layers': 1,
        'units': 500,
        'bidirectional_lstm': True,
        'epochs': 50,
        'loss_function': 'binary_crossentropy',
        'optimizer': 'Adam',  # SGD, Adam, Adamax, etc
        'activation': 'softmax',  # softmax, relu, etc
        'evaluation_set': 'dev',  # dev or test
    }

    if create_file:
        with open('lstm_config.json', 'x', encoding='utf-8') as config_file:
            json.dump(default_config, config_file, indent=4)

    return default_config


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments using the argparse parser"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--config_file',
        default='plm_config.json',
        type=str,
        help='Config file to use (default plm_config.json)',
    )

    parser.add_argument(
        '-x',
        '--create_config',
        action='store_true',
        help='Creates a config file with default parameters (plm_config.json)',
    )

    parser.add_argument(
        '-o',
        '--output_folder',
        default='./model/',
        help='Where to output the model when fine-tuned (default ./model/)',
    )

    args = parser.parse_args()

    return args


def read_embeddings(embeddings_file: str) -> dict[str, np.NDArray[np.float64, ...]]:
    """Read in word embeddings from file and save as numpy array"""
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    """Get embedding matrix given vocab and the embeddings"""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word 'the'
    embedding_dim = len(emb['the'])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings for feeding to embedding layer
    return embedding_matrix


def create_model(
    Y_train,
    emb_matrix,
    config: dict,
) -> Sequential:
    """Create the Keras model to use. Only uses voc if voc is passed"""
    # Define settings, you might want to create cmd line args for them
    loss_function = config['loass_function']
    match config['optimizer']:
        case 'SGD':
            optim = SGD(learning_rate=config['learning_rate'])
        case 'Adam':
            optim = Adam(learning_rate=config['learning_rate'])
        case 'Adamax':
            optim = Adamax(learning_rate=config['learning_rate'])

    num_labels = len(set(Y_train))

    # Now build the model
    model = Sequential()

    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    model.add(
        Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=Constant(emb_matrix),
            trainable=False,
        )
    )

    # add LSTM layers
    if not config['layers'] > 1:
        for _ in range(config['layers'] - 1):
            if config['bidirectional']:
                model.add(
                    Bidirectional(
                        LSTM(
                            units=config['units'],
                            dropout=config['dropout'],
                            return_state=True,
                        )
                    )
                )
            else:
                model.add(
                    LSTM(
                        units=config['units'],
                        dropout=config['dropout'],
                        return_state=True,
                    )
                )

    if config['bidirectional']:
        model.add(Bidirectional(LSTM(units=700, dropout=config['dropout'])))
    else:
        model.add(LSTM(units=2, dropout=config['dropout']))

    # Ultimately, end with dense layer with softmax
    model.add(Dense(units=num_labels, activation=config['activation']))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, config: dict):
    """Train the model here. Note the different settings you can experiment with!"""
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    batch_size = config['batch_size']
    epochs = config['epochs']
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor='loss'
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # Finally fit the model to our data
    model.fit(
        X_train,
        Y_train,
        verbose=verbose,
        epochs=epochs,
        callbacks=[callback],
        batch_size=batch_size,
        validation_data=(X_dev, Y_dev),
    )
    # Print final accuracy for the model (clearer overview)
    return model


def main():
    """Main function to train and test neural network given cmd line arguments"""
    args = parse_args()

    if args.create_config:
        create_default_config(True)
        return

    config = parse_config(args.config_file, create_default_config())

    # Read in the data and embeddings
    data = load_data(
        config['data_dir'],
        OffensiveWordReplaceOption.from_str(config['replace_option']),
        config['preprocessed'],
    )
    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(
        data['train'].documents + data['dev'].documents
    )
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(
        data['train'].labels
    )  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(data['dev'].labels)

    model = create_model(data['train'].labels, emb_matrix, config)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in data['train'].documents])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in data['dev'].documents])).numpy()

    # Train the model
    model = train_model(
        model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, config['batch_size']
    )

    model.save(args.output_folder)


if __name__ == '__main__':
    main()
