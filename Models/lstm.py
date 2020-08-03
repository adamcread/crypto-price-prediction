import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython
import IPython.display


def lstm(df):
    class WindowGenerator():
        def __init__(self, input_width, label_width, shift,
                    train_df, val_df, test_df,
                    label_columns=None):

            # Work out the label column indices.
            self.label_columns = label_columns
            if label_columns is not None:
                self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
            
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

             # Store the raw data.
            self.train_df = np.array(train_df)
            self.val_df = np.array(val_df)
            self.test_df = np.array(test_df)

            # Work out the window parameters.
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift

            self.total_window_size = input_width + shift

            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]

            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        def split_window(self, features):
            inputs = features[:, self.input_slice, :]
            labels = features[:, self.labels_slice, :]
            if self.label_columns is not None:
                labels = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

            # Slicing doesn't preserve static shape information, so set the shapes
            # manually. This way the `tf.data.Datasets` are easier to inspect.
            inputs.set_shape([None, self.input_width, None])
            labels.set_shape([None, self.label_width, None])

            return inputs, labels

        def make_dataset(self, data):
            data = np.array(data, dtype=np.float32)
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32,)

            ds = ds.map(self.split_window)

            return ds

        def train(self):
            return self.make_dataset(self.train_df)

        def val(self):
            return self.make_dataset(self.val_df)

        def test(self):
            return self.make_dataset(self.test_df)

        def example(self):
            """Get and cache an example batch of `inputs, labels` for plotting."""
            result = getattr(self, '_example', None)
            if result is None:
                # No example batch was found, so get one from the `.train` dataset
                result = next(iter(self.train()))
                # And cache it for next time
                self._example = result
            return result

        def plot(self, model=None, plot_col='Close', max_subplots=3):
            inputs, labels = self.example()
            plt.figure(figsize=(12, 8))
            plot_col_index = self.column_indices[plot_col]
            max_n = min(max_subplots, len(inputs))
            for n in range(max_n):
                plt.subplot(3, 1, n+1)
                plt.ylabel(f'{plot_col} [normed]')
                plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                        label='Inputs', marker='.', zorder=-10)

                if self.label_columns:
                    label_col_index = self.label_columns_indices.get(plot_col, None)
                else:
                    label_col_index = plot_col_index

                if label_col_index is None:
                    continue

                plt.scatter(self.label_indices, labels[n, :, label_col_index],
                            edgecolors='k', label='Labels', c='#2ca02c', s=64)
                if model is not None:
                    predictions = model(inputs)

                    plt.scatter(range(self.total_window_size-1), predictions[n, :, label_col_index],
                                marker='X', edgecolors='k', label='Predictions',
                                c='#ff7f0e', s=64)

                if n == 0:
                    plt.legend()

            plt.xlabel('Time [h]')

            plt.show()

        def __repr__(self):
            return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])

    # split the data into training, validation, and test sets
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    # normalize the data so properly scaled innit
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    print(train_df)
    print(len(train_df))

    # ? window data by using a the previous week of data 
    # ? window - input-width: 7, offset:1, label-width:1

    win = WindowGenerator(input_width=14, label_width=1, shift=1, 
                        train_df=train_df, val_df=val_df, test_df=test_df)

    example_window = tf.stack([np.array(train_df[100:100+win.total_window_size]),
                           np.array(train_df[1000:1000+win.total_window_size]),
                           np.array(train_df[1750:1750+win.total_window_size])])

    example_inputs, example_labels = win.split_window(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'labels shape: {example_labels.shape}')
    # full_window = tf.stack([np.array(train_df[i:i+win.total_window_size]) for i in 
    #                         range(0, len(train_df)-40, win.total_window_size)])


    # win.train()

    MAX_EPOCHS = 20
    def compile_and_fit(model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

        validation = window.val()
        training = window.train()

        history = model.fit(training, epochs=MAX_EPOCHS,
                            validation_data=validation,
                            callbacks=[early_stopping])

        return history
    
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(227, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(lstm_model, win)

    # IPython.display.clear_output()
    # val_performance['LSTM'] = lstm_model.evaluate(win.val)
    # performance['LSTM'] = lstm_model.evaluate(win.test, verbose=0)

    win.plot(lstm_model)


    

bitcoin_data = pd.read_csv('Data/bitcoin_data.csv')
bitcoin_data = bitcoin_data.drop(labels=['Date', 'Open', 'High', 'Low', 'Volume', 'Market Cap'], axis=1)
lstm(bitcoin_data)