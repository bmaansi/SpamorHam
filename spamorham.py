import tensorflow_decision_forests as tfdf

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math

from IPython.display import display

dataset_df = pd.read_csv("spambase.csv")

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


# Specify the model.
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model.
model_1.fit(train_ds)

user_input = input("Enter email: ")

email = str(user_input)

email_data = {
  'word_freq_make': email.count('make'),
  'word_freq_address': email.count('address'),
  'word_freq_all': email.count('all'),
  'word_freq_3d': email.count('3d'),
  'word_freq_our': email.count('our'),
  'word_freq_over': email.count('over'),
  'word_freq_remove': email.count('remove'),
  'word_freq_internet': email.count('internet'),
  'word_freq_order': email.count('order'),
  'word_freq_mail': email.count('mail'),
  'word_freq_receive': email.count('receive'),
  'word_freq_will': email.count('will'),
  'word_freq_people': email.count('people'),
  'word_freq_report': email.count('report'),
  'word_freq_addresses': email.count('addresses'),
  'word_freq_free': email.count('free'),
  'word_freq_business': email.count('business'),
  'word_freq_email': email.count('email'),
  'word_freq_you': email.count('you'),
  'word_freq_credit': email.count('credit'),
  'word_freq_your': email.count('your'),
  'word_freq_font': email.count('font'),
  'word_freq_000': email.count('000'),
  'word_freq_money': email.count('money'),
  'word_freq_hp': email.count('hp'),
  'word_freq_hpl': email.count('hpl'),
  'word_freq_george': email.count('george'),
  'word_freq_650': email.count('650'),
  'word_freq_lab': email.count('lab'),
  'word_freq_labs': email.count('labs'),
  'word_freq_telnet': email.count('telnet'),
  'word_freq_857': email.count('857'),
  'word_freq_data': email.count('data'),
  'word_freq_415': email.count('415'),
  'word_freq_85': email.count('85'),
  'word_freq_technology': email.count('technology'),
  'word_freq_1999': email.count('1999'),
  'word_freq_parts': email.count('parts'),
  'word_freq_pm': email.count('pm'),
  'word_freq_direct': email.count('direct'),
  'word_freq_cs': email.count('cs'),
  'word_freq_meeting': email.count('meeting'),
  'word_freq_original': email.count('orginal'),
  'word_freq_project': email.count('project'),
  'word_freq_re': email.count('re'),
  'word_freq_edu': email.count('edu'),
  'word_freq_table': email.count('table'),
  'word_freq_conference': email.count('conference'),
  'char_freq_;': email.count(';'),
  'char_freq_(': email.count('('),
  'char_freq_[': email.count('['),
  'char_freq_!': email.count('!'),
  'char_freq_$': email.count('$'),
  'char_freq_#': email.count('#'),
  'capital_run_length_average': 1,
  'capital_run_length_longest': 1,
  'capital_run_length_total': 1,
}

email_df = pd.DataFrame([email_data])
email_predictions = tfdf.keras.pd_dataframe_to_tf_dataset(email_df, task= tfdf.keras.Task.CLASSIFICATION)
predictions = model_1.predict(email_predictions)

predicted_label = int(predictions[0][0])

if predicted_label == 0:
    print("The email is classified as 'ham'.")
else:
    print("The email is classified as 'spam'.")
