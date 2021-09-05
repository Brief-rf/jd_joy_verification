import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
import sys
import flask
import io
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from model import brief_net
from tensorflow.python.keras.backend import set_session
from gevent import pywsgi

sess = tf.Session()
graph = tf.get_default_graph()
app = flask.Flask(__name__)
def load_model():
    global model
    model = brief_net(input_shape=(140, 360, 3), output_shape=1)
    set_session(sess)
    try:
        model.load_weights("trained_weights.h5")
    except Exception as e:
        print(e)
        sys.exit(0)
    global graph
    graph = tf.get_default_graph()

def prepare_image(image):
    image = np.array(image) / 255.
    image = image[:,:,:3]
    image = (np.expand_dims(image, axis=0))
    return image
@app.route("/predict", methods=["POST"])
def predict():
    global sess, graph
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image)
            with graph.as_default():
                set_session(sess)
                preds = model.predict(image)
            data["predictions"] = round(preds.tolist()[0][0], 3)
            data["success"] = True
    return flask.jsonify(data)

if __name__ == "__main__":
    port = 7000
    try:
        port = int(sys.argv[1])
    except Exception as e:
        print(e)
    print("* Loading Keras model and Flask starting server...")
    load_model()
    print("* Model loaded successfully!")
    server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    print("* Listening on http://0.0.0.0:{}/predict".format(port))
    server.serve_forever()