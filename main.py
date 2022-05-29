# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data():
    data = pd.read_csv("data/abalone.data") #ucitavanje podataka iz datoteke
    data_visualization(data) #vizualizacija podataka

    one_hot_gender = pd.get_dummies(data['Sex'], prefix='Sex')
    data = data.drop('Sex', axis=1)
    data = data.join(one_hot_gender) #menjanje kolone 'Sex' one hot vrednostima

    X = data.drop('Rings', axis=1).values.astype(float) #podela podataka na ulazne
    Y = data['Rings'].values.astype(float) # i izlazne

    scale = StandardScaler() #skaliranje podataka
    X[:, 0:7] = scale.fit_transform(X[:, 0:7]) #one hot vrednosti se ne skaliraju

    return X, Y


def data_visualization(data):
    data_table_file = open("data/abalone.table", "w")
    data_table_file.write(tabulate(data, headers='keys', tablefmt='psql')) #tabelarni prikaz podataka

    data.hist(figsize=(20, 20), grid=True, layout=(2, 4), bins=30) #histogram
    plt.show()

    sns.countplot(x='Sex', data=data) #distribucija polova
    plt.show()


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
        model.add(tf.keras.layers.Dense(
            30,
            activation='leaky_relu',
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform'
        ))
        model.add(tf.keras.layers.Softmax(-1))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:
        model.add(tf.keras.layers.Dense(
            1,
            activation='linear',
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform'
        ))
        loss = "mean_squared_error"
        metrics = ['mean_absolute_error']

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=metrics
    )
    return model


def visualize_prediction(Y_train, Y_test, Y_predict_softmax, Y_predict_linear):
    df = pd.DataFrame()
    df['Y_train'] = pd.Series(Y_train)
    df['Y_test'] = pd.Series(Y_test)
    df['Y_predict_softmax'] = pd.Series(Y_predict_softmax.reshape(-1))
    df['Y_predict_linear'] = pd.Series(Y_predict_linear.reshape(-1))
    df.hist(figsize=(20, 20), grid=True, layout=(2, 2), bins=30)
    plt.show()


def main():
    X, Y = load_and_preprocess_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    softmax_model = create_model([16, 8, 6, 4, 2], 'softmax')
    softmax_model.build(X_train.shape)
    softmax_model.summary()
    softmax_model.fit(X_train, Y_train, epochs=600, batch_size=256)
    softmax_model.evaluate(X_test, Y_test)
    Y_predict_softmax = np.argmax(softmax_model.predict(X_test), -1, keepdims=True)

    linear_model = create_model([32, 16, 16, 8, 8, 4], 'linear')
    linear_model.build(X_train.shape)
    linear_model.summary()
    linear_model.fit(X_train, Y_train, epochs=1200, batch_size=1024)
    linear_model.evaluate(X_test, Y_test)
    Y_predict_linear = linear_model.predict(X_test)

    visualize_prediction(Y_train, Y_test, Y_predict_softmax, Y_predict_linear)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/