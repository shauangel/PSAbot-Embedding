# word embeddings
import tensorflow as tf
import tensorflow_hub as hub

embed = hub.KerasLayer("embeds/Wiki-words-250_2",
                           input_shape=[],
                           dtype=tf.string,
                           trainable=True,
                           name="Word_Embedding_Layer")
user_q_vector = embed(["Cat is a great pet.", "Dog is a great pet.", "But I want a hamster."])

print(user_q_vector)
