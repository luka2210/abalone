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


def create_model(hidden_layers, output_layer='linear'):
    model = tf.keras.Sequential()

    for layer_size in hidden_layers:
        model.add(tf.keras.layers.Dense(
            layer_size,
            activation='leaky_relu',
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform'
        ))

    if output_layer == 'softmax':
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
        model.add(tf.keras.layers.Dense(
            30,
            activation='leaky_relu',
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform'
        ))
        model.add(tf.keras.layers.Softmax(-1))
    else:
        loss = "mean_squared_error"
        metrics = ['mean_absolute_error']
        model.add(tf.keras.layers.Dense(
            1,
            activation='linear',
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform'
        ))

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=metrics
    )

    return model


def main():
    X, Y = load_and_preprocess_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    softmax_model = create_model([16, 8, 6, 4, 2], 'softmax')
    softmax_model.fit(X_train, Y_train, epochs=600, batch_size=256)
    softmax_model.evaluate(X_test, Y_test)

    linear_model = create_model([32, 16, 16, 8, 8, 4], 'linear')
    linear_model.fit(X_train, Y_train, epochs=600, batch_size=256)
    linear_model.evaluate(X_test, Y_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/