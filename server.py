from flask import Flask, request, jsonify
from . import models
import numpy as np

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def get_embeddings():
    data = request.get_json(force=True)
    embeds = models.get_embeddings(data['doc_list'])
    return jsonify(embeds.numpy().tolist())

if __name__ == '__main__':
    app.run(port=10003, debug=True)
