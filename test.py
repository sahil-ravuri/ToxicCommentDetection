# %% [markdown]
# # 0. Install Dependencies and Bring in Data

# %%
import os
import pandas as pd
import tensorflow as tf
import numpy as np

# %%
df = pd.read_csv(os.path.join('train.csv'))

# %%
df.head()

# %% [markdown]
# # 1. Preprocess

# %%


# %%
from tensorflow.keras.layers import TextVectorization

# %%
X = df['comment_text']
y = df[df.columns[2:]].values

# %%
MAX_FEATURES = 200000 # number of words in the vocab

# %%
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')

# %%
vectorizer.adapt(X.values)

# %%
vectorized_text = vectorizer(X.values)

# %%
#MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottlenecks

# %%
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

# %% [markdown]
# # 2. Create Sequential Model

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

# %%
model = Sequential()
# Create the embedding layer 
model.add(Embedding(MAX_FEATURES+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer 
model.add(Dense(6, activation='sigmoid'))

# %%
model.compile(loss='BinaryCrossentropy', optimizer='Adam')

# %%
model.summary()

# %%
history = model.fit(train, epochs=1, validation_data=val)

# %%
from matplotlib import pyplot as plt

# %%
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()

# %% [markdown]
# # 3. Make Predictions

# %%
input_text = vectorizer('You freaking suck! I am going to hit you.')

# %%
res = model.predict(input_text)

# %%
(res > 0.5).astype(int)

# %%
batch_X, batch_y = test.as_numpy_iterator().next()

# %%
(model.predict(batch_X) > 0.5).astype(int)

# %%
res.shape

# %% [markdown]
# # 4. Evaluate Model

# %%
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

# %%
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

# %%
for batch in test.as_numpy_iterator(): 
    # Unpack the batch 
    X_true, y_true = batch
    # Make a prediction 
    yhat = model.predict(X_true)
    
    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

# %%
print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

# %% [markdown]
# # 5. Test and Gradio

# %%

# %%
import tensorflow as tf
import gradio as gr

# %%
model.save('toxicity.h5')

# %%
model = tf.keras.models.load_model('toxicity.h5')

# %%
input_str = vectorizer('hey i freaken hate you!')

# %%
res = model.predict(np.expand_dims(input_str,0))

# %%
res

# %%
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text

# %%
interface = gr.Interface(fn=score_comment, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

# %%
interface.launch(share=True)


