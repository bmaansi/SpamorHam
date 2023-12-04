import tensorflow_decision_forests as tfdf
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import re
import sys
from matplotlib import pyplot as plt
import dtreeviz
from IPython import display

#Set seed to 6789 to ensure the same training/testing split for dataset and same models when analyzing

import logging
logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)

display.set_matplotlib_formats('retina') # generate hires plots

np.random.seed(6789)  # reproducible plots/data for explanatory reasons

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

def split_dataset(dataset, test_ratio=0.30, seed=6789):
  """Splits a panda dataframe in two."""
  np.random.seed(seed)
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

# Specify the model.
model_1 = tfdf.keras.RandomForestModel(verbose=2, random_seed=6789)

# Train the model.
model_1.fit(train_ds)

model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

# model_1.save("/tmp/my_saved_model")

tfdf.model_plotter.plot_model_in_colab(model_1, tree_idx=0, max_depth=3)

#Build, Train, and evaluate code ends briefly, Visualizing code starts

# IMPORTANT TIP: A value of 1 means spam, a value of 0 means ham

# Visualizing Trees code starts here

# Tell dtreeviz about training data and model
spam_features = [f.name for f in model_1.make_inspector().features()]
viz_model_1 = dtreeviz.model(model_1,
                           tree_index=3,
                           X_train=train_ds_pd[spam_features],
                           y_train=train_ds_pd[label],
                           feature_names=spam_features,
                           target_name=label,
                           class_names=classes)

# PUT BREAKPOINTS AFTER v.show() at the next line of real code

v = viz_model_1.view(scale=5) # // Best General Overview for our model, LIKELY USE
v.show()

# Root node of the tree, mainly useless for our plot, doesn't show much value, just leave commented or delete
# v = viz_model_1.view(depth_range_to_display=[0,0], scale=10)
# v.show()

# Second level of the tree, mainly useless for our plot, doesn't show much value, just leave commented or delete
# v = viz_model_1.view(depth_range_to_display=[1,1], scale=5)
# v.show()

# Simpler versions of the tree display // USE THIS most likely
# v = viz_model_1.view(fancy=False, scale=.75)
# v.show()

# Left to right verision, pretty much useless for our plot, just leave commented or delete
# v = viz_model_1.view(orientation='LR', scale=.75)
# v.show()

# Bar chart version instead of pie chart // Maybe want to stick to Pie chart so maybe not use?
# v = viz_model_1.view(leaftype='barh', scale=.75)
# v.show()

# Examine the number of training data instances that are grouped into each leaf node // NOT WORKING
# v = viz_model_1.leaf_sizes(figsize=(5,1.5))
# v.show()

# How the classifier makes a decision for a specific instance, highlight the path from the root to the leaf pursued by the classifier to make the 
# prediction for that instance. "show_just_path=True" function shows a simpler more readable picture of the path // USE ONE OR BOTH OF THESE FOR SURE
x = train_ds_pd[spam_features].iloc[50]
v = viz_model_1.view(x=x, scale=5)
v.show()
v = viz_model_1.view(x=x, show_just_path=True, scale=5)
v.show()

# Prints the path in english word form, however doesn't seem to link up with the images' path.
# print(viz_model_1.explain_prediction_path(x=x))


# NOT WORKING
# v = viz_model_1.ctree_feature_space(features=['word_freq_make'], show={'splits','legend'}, figsize=(5,1.5))
# v.show()

# Visualizing Trees code ends here




# Build, Train, and evaluate code begins again
model_1.summary()

# The input features
model_1.make_inspector().features()

# The feature importances
model_1.make_inspector().variable_importances()

model_1.make_inspector().evaluation()


model_1.make_inspector().training_logs()



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

plt.show()

def calculate_run_length_features(email) :
  runs = re.findall(r"[A-Z]+", email)

  capital_run_length_total = sum(len(run) for run in runs)

  capital_run_length_longest = max(len(run) for run in runs) if runs else 0

  capital_run_length_average = capital_run_length_total / len(runs) if runs else 0
 
  return capital_run_length_total, capital_run_length_longest, capital_run_length_average

while 1:

  #user_input = input("Enter email: ")
  print("Enter email: ")
  user_input = sys.stdin.read() 
  email = str(user_input)

  total, longest, average = calculate_run_length_features(email)

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
    "capital_run_length_average": average,
    "capital_run_length_longest": longest,
    "capital_run_length_total": total,
  }

  email_df = pd.DataFrame([email_data])
  email_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(email_df)

  prediction = model_1.predict(email_dataset, verbose=0)
  p = np.round(prediction).astype(int)
  if p == 1:
    print("Email is spam")
  else:
    print("Email is ham")

  b = input("Check another email? (Y/N): ")
  if b == 'N' or b == 'n':
    break
