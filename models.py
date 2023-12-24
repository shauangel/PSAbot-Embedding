import tensorflow as tf
import tensorflow_hub as hub

embed = hub.KerasLayer("embeds/Wiki-words-250_2",
                           input_shape=[],
                           dtype=tf.string,
                           trainable=True,
                           name="Word_Embedding_Layer")

def get_embeddings(doc_list):
    vector = embed(doc_list)

    # print(vector)
    return vector

def get_similarity(a, b):
    return
