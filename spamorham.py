import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

from IPython.display import display

# Download the dataset
# wget -q https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins.csv -O /tmp/penguins.csv

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("spambase.csv")

# Display the first 3 examples.
dataset_df.head(3)

# Encode the categorical labels as integers.
#
# Details:
# This stage is necessary if your classification label is represented as a
# string since Keras expects integer classification labels.
# When using `pd_dataframe_to_tf_dataset` (see below), this step can be skipped.

# Name of the label column.
label = "spam"

classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)

# Split the dataset into a training and a testing dataset.

def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

# %set_cell_height 300

# Specify the model.
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model.
model_1.fit(train_ds)

model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

model_1.save("/tmp/my_saved_model")

tfdf.model_plotter.plot_model_in_colab(model_1, tree_idx=0, max_depth=3)

# %set_cell_height 300
model_1.summary()

# The input features
model_1.make_inspector().features()

# The feature importances
model_1.make_inspector().variable_importances()

model_1.make_inspector().evaluation()

#%set_cell_height 150
model_1.make_inspector().training_logs()

import matplotlib.pyplot as plt

logs = model_1.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")



user_input = input("Enter email: ")

email = str(user_input)



email_data = {
  "word_freq_make": email.lower().count('make'),
  "word_freq_address": email.lower().count('address'),
  "word_freq_all": email.lower().count('all'),
  "word_freq_3d": email.lower().count('3d'),
  "word_freq_our": email.lower().count('our'),
  "word_freq_over": email.lower().count('over'),
  "word_freq_remove": email.lower().count('remove'),
  "word_freq_internet": email.lower().count('internet'),
  "word_freq_order": email.lower().count('order'),
  "word_freq_mail": email.lower().count('mail'),
  "word_freq_receive": email.lower().count('receive'),
  "word_freq_will": email.lower().count('will'),
  "word_freq_people": email.lower().count('people'),
  "word_freq_report": email.lower().count('report'),
  "word_freq_addresses": email.lower().count('addresses'),
  "word_freq_free": email.lower().count('free'),
  "word_freq_business": email.lower().count('business'),
  "word_freq_email": email.lower().count('email'),
  "word_freq_you": email.lower().count('you'),
  "word_freq_credit": email.lower().count('credit'),
  "word_freq_your": email.lower().count('your'),
  "word_freq_font": email.lower().count('font'),
  "word_freq_000": email.lower().count('000'),
  "word_freq_money": email.lower().count('money'),
  "word_freq_hp": email.lower().count('hp'),
  "word_freq_hpl": email.lower().count('hpl'),
  "word_freq_george": email.lower().count('george'),
  "word_freq_650": email.lower().count('650'),
  "word_freq_lab": email.lower().count('lab'),
  "word_freq_labs": email.lower().count('labs'),
  "word_freq_telnet": email.lower().count('telnet'),
  "word_freq_857": email.lower().count('857'),
  "word_freq_data": email.lower().count('data'),
  "word_freq_415": email.lower().count('415'),
  "word_freq_85": email.lower().count('85'),
  "word_freq_technology": email.lower().count('technology'),
  "word_freq_1999": email.lower().count('1999'),
  "word_freq_parts": email.lower().count('parts'),
  "word_freq_pm": email.lower().count('pm'),
  "word_freq_direct": email.lower().count('direct'),
  "word_freq_cs": email.lower().count('cs'),
  "word_freq_meeting": email.lower().count('meeting'),
  "word_freq_original": email.lower().count('original'),
  "word_freq_project": email.lower().count('project'),
  "word_freq_re": email.lower().count('re'),
  "word_freq_edu": email.lower().count('edu'),
  "word_freq_table": email.lower().count('table'),
  "word_freq_conference": email.lower().count('conference'),
  "char_freq_semicolon": email.lower().count(';'),
  "char_freq_parenthesis": email.lower().count('('),
  "char_freq_bracket": email.lower().count('['),
  "char_freq_exclamation": email.lower().count('!'),
  "char_freq_dollarsign": email.lower().count('$'),
  "char_freq_hash": email.lower().count('#'),
  "capital_run_length_average": 1,
  "capital_run_length_longest": 1,
  "capital_run_length_total": 1,
}

email_df = pd.DataFrame([email_data])
email_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(email_df)

prediction = model_1.predict(email_dataset, verbose=0)

print(prediction)
