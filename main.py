# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():
    data = pd.read_csv("data/abalone.data")

    one_hot_gender = pd.get_dummies(data['Sex'], prefix='Sex')
    data = data.drop('Sex', axis=1)
    data = data.join(one_hot_gender)

    X = data.drop('Rings', axis=1).values.astype(float)
    Y = data['Rings'].values.astype(float)

    scale = StandardScaler()
    X[:, 0:7] = scale.fit_transform(X[:, 0:7])

    return X, Y


def create_model(input_dim, hidden_layers):
    model = tf.keras.Sequential()

    for layer_size in hidden_layers:
        model.add(tf.keras.layers.Dense(
            layer_size,
            input_dim=input_dim,
            activation='leaky_relu',
            kernel_initializer='glorot_uniform',
            bias_initializer="glorot_uniform",
            #kernel_regularizer="l1"
        ))
    model.add(tf.keras.layers.Softmax(-1))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X, Y = load_and_preprocess_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    model = create_model(X.shape[1], [16, 8, 6, 4, 2, 30])
    model.summary()
    model.fit(X_train, Y_train, epochs=600, batch_size=512)
    model.evaluate(X_test, Y_test)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/