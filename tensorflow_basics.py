import tensorflow as tf
from matplotlib import pyplot
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.9:  # Experiment with changing this value
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True


early_stopping = EarlyStopping(patience=5, monitor='loss', verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='loss',
                              min_lr=0.001,
                              patience=5,
                              mode='min',
                              verbose=1)

model_checkpoint = ModelCheckpoint(monitor='loss',
                                   filepath='./model-best-kaggle-tl.h5',
                                   save_best_only=True)

mycallback = MyCallback()


callbacks = [
    early_stopping,
    reduce_lr,
    model_checkpoint,
    mycallback
]

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=10, callbacks=[callbacks])


# evaluate the model
_, test_acc = model.evaluate(test_images, test_labels)
_, train_acc = model.evaluate(training_images, training_labels)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))




# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['loss'], label='test')
pyplot.legend()
pyplot.show()




classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
