#%% [markdown]
# ## Exercise 3
# In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. You should stop training once the accuracy goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.
# 
# I've started the code for you -- you need to finish it!
# 
# When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
# 

#%%
import tensorflow as tf

# YOUR CODE STARTS HERE

# YOUR CODE ENDS HERE

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images.reshape(training_images.shape[0], 28,28,1)
training_images = training_images / 255.0

test_images = test_images.reshape(test_images.shape[0], 28,28,1)
test_images = test_images / 255.0


model = tf.keras.models.Sequential([
    # YOUR CODE STARTS HERE
    tf.keras.layers.Conv2D(256, (3,3), input_shape = (28, 28, 1), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])



model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(training_images, training_labels, epochs = 10)
test_loss = model.evaluate(test_images, test_labels)


