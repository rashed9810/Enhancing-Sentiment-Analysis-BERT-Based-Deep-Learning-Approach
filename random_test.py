import torch
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification

model_path = 'bert_epoch_2_3/'
model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Generate random sentences
random_sentence = [
    "I love the sunshine on a bright day.",
    "The cat chased the mouse across the room.",
    "Pizza is my favorite food.",
    "The concert was absolutely amazing.",
    "Learning new things is always exciting.",
    "I'm working on a new project using a no-code platform for development.",
    "She was feeling really happy after receiving the good news.",
    "This topic is not-relevant to the discussion we're having.",
    "He became angery when his computer crashed right before the presentation.",
    "The movie's graphic content left me feeling a mix of disgust and anger.",
    "The unexpected gift left her happy and surprised at the same time.",
    "His sad and angry expression revealed how deeply he was affected by the news.",
    "The magician's trick left the audience in awe and surprise.",
    "The old man sat alone on the park bench, lost in memories of his late wife.",
    "As the magician pulled back the curtain, the audience gasped in astonishment at the empty stage.",
    "After years of promises, they betrayed my trust once again, leaving me seething with a mix of disappointment and rage.",
    "She looked at the old photographs, tears streaming down her face as she remembered the days that would never come back."
]

# tokenize and encode random text
encoded_random_text = tokenizer.batch_encode_plus(
    random_sentence,
    add_special_tokens = True,
    return_attention_mask = True,
    padding = 'max_length',
    truncation = True,
    max_length = 256,
    returns_tensors = 'tf'    
) 
input_ids_random = encoded_random_text['input_ids']
attention_masks_random = encoded_random_text['attention_mask']

input_ids_random = tf.convert_to_tensor(input_ids_random)
attention_masks_random = tf.convert_to_tensor(attention_masks_random)

# using prediction for pre-trained model
random_prediction = model([input_ids_random, attention_masks_random], training = False)[0]

# convert prediction to label
possible_labels = ["nocode", "happy", "not-relevant", "angery", "disgust|angry", "happy|surprise", "sad|angry", "surprise"  ]
random_prediction_flat = np.argmax(random_prediction, axis = 1)
predicted_categories = [possible_labels[label] for label in random_prediction_flat]


# print predicted categories for random text
for sentence, predicted_category in zip(random_sentence, predicted_categories):
    print(f"Sentence: {sentence}")
    print(f"Predicted Category: {predicted_category}\n")

# data exploration and analysis
