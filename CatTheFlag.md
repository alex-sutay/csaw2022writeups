## Initial Process
The only hints we got was that this is an intro to Neural Networks and Data Analysis challenge.
The linked website evaluates a model, so I knew I had to train something.
The download contained 3 pickle files: X, y, and x_test. 
Interestingly there was no y_test.

## Examining the data
It's common in ML to divide your data set. 
Typically you have X as your inputs and y as your expected outputs.
Then you cut off a portion of X and y to be your test data and then train on the rest.
Here we have the training X and y, x_test, but no y_test.
More on that later.
For now, I decided to see what the data actually is. 
Examining the shape, it's a series 2D arrays, 80x80, each containing a set of 3 values, or in other words each data point is 80x80x3.
When I see data with 3 points, I immediately think RGB, especially with the other dimension being equal.
This was likely an 80 by 80 image with RGB values. 
I set it to display the top 5 and sure enough, it's pictures of cats and dogs. 
Checking the y values, I see that it's all 1s or 0s.
The 1s corresponded to dogs and the 0s corresponded to cats.
So the problem was to be a simple classifier.

## Building the classifier
I started by finding a basic image classifier NN in tensorflow and setting it to run on the data. 
It worked alright, but stalled out at about 80% accuracy.
I tried this algorithm on the site provided, and it claimed I only had ~50% accuracy. 
Something was wrong.
Examining the training data, I found that it was 80% cats. 
So the classifier was getting stuck assuming everything was cats, stalling at 80% accuracy.
To confirm, I trained a model to definitely assume cat every time and one to assume dog and tried each on the site.
They both reported about 50% accuracy.
This means that the test set on the site was half dog and half cat, and my original model assumed all cat, and got 80% training accuracy and 50% test accuracy.
To fix this, I just beefed up my model, using a more advanced loss alogrithm and better layers, and training longer. 
Now my model approached 95% accuracy.
(As a note, for somewhere I got my wires crossed and the site reported 5% accuracy, which just means I was classifying 1s as 0s and 0s as 1s.
This is fine because all I have to do is reverse the output I get and 0% accuracy turns into 100% accuracy)
With an effective model, now all I had to do was find the flag.

## Finding the flag
I check the x_test set and there are 288 entries.
This is suspicious because 288 is evenly divisible by 8; it could be bytes.
I set a new python file to load the model and classify each entry. 
Then I take 8 classifications at a time, treat it as binary representing a byte, and look up the ascii value of that byte.
This gave me a messy but mostly legible flag.
To fix it up, I retrained to get better accuracy and a better picture of the flag. 
My best model had 100% training accuracy and output the flag perfectly.

## Code
Model training:
```
import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescal                                                                                                             ing

with open('X.pkl', 'rb') as f:
        X = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
with open('y.pkl', 'rb') as f:
        y = pickle.load(f)

X_zero = np.array([X[i] for i in range(len(y)) if y[i] == 0])
X_one = [X[i] for i in range(len(y)) if y[i] == 1]

new_X = np.append(X_one, X_zero[:len(X_one)], axis=0)
new_y = np.append(np.ones(len(X_one)), np.zeros(len(X_one)))

model = Sequential([
    Rescaling(1./255, input_shape=(80, 80, 3)),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2)
    ])

model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])

print(model.summary())

model.fit(X, y, batch_size=32, epochs=20, shuffle=True)

model.save('model.h1')
```
Getting the flag:
```
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.h1')

with open('X_test.pkl', 'rb') as f:
    test = pickle.load(f)

print(test.shape)
res = model1.predict(test)

res = ''.join('0' if val[0] > val[1] else '1' for val in res)

sol = ''
for i in range(0, len(res1), 8):
    b = res1[i:i+8]
    print(f'{b}:\t{chr(int(b1, 2))}')
    sol += chr(int(b, 2))

print(sol)
```
