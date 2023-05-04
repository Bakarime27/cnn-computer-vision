# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import cv2 as cv
import gradio as gr
from gradio.networking import INITIAL_PORT_VALUE, LOCALHOST_NAME

print(tf.__version__)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Load model
model = load_model('static/xception_hypermodel.h5', custom_objects={"f1_m": f1_m})
breed_labels = ['Australian_terrier', 'Border_collie', 'Chihuahua', 'English_setter',
 'French_bulldog', 'German_shepherd', 'Labrador_retriever', 'Rottweiler',
 'Shih', 'Siberian_husky', 'Staffordshire_bullterrier', 'Yorkshire_terrier',
 'beagle', 'cocker_spaniel', 'golden_retriever']

gradio_desc = 'This interface gives the possibility of determining a dog breed directly from a photo.'\
                'Just drag your image into the left frame and wait a few seconds !\n'\
                'The model was trained on the 15 most popular dog breeds in France'
                
gradio_article = ('<h3>General informations</h3><p>The objective of this Notebook is to detail the implementation'+
    ' of a dog breed detection algorithm on a photo, in order to speed up the work of indexing in a database.</p>'+
    '<h3>The constraints imposed</h3><p><ul><li>Pre-process the images with specific techniques (e.g. Whitening, equalization,'+
    ' possibly modification of the size of the images).</li>'+
    '<li>Perform data augmentation (mirroring, cropping ...).</li>'+
    '<li>Implementation of 2 approaches to the use of CNNs :'+
    '<ul><li>Create a CNN neural network from scratch by optimizing the parameters.</li>'+
    '<li>Use the transfer learning and thus use an already trained network.</li>'+
    '<li>Fine-tuning of the pre-trained model</li></ul></li></ul></p>')

# Define the full prediction function
def breed_prediction(inp):
    # Convert to RGB
    img = cv.cvtColor(inp,cv.COLOR_BGR2RGB)
    # Resize image
    dim = (299, 299)
    img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
    # Equalization
    img_yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
    img_equ = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
    # Apply non-local means filter on test img
    dst_img = cv.fastNlMeansDenoisingColored(
        src=img_equ,
        dst=None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21)

    # Convert modified img to array
    img_array = tf.keras.preprocessing.image.img_to_array(dst_img)
    
    # Apply preprocess Xception
    img_array = img_array.reshape((-1, 299, 299, 3))
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    
    # Predictions
    prediction = model.predict(img_array).flatten()
    
    #return prediction
    return {breed_labels[i]: float(prediction[i]) for i in range(len(breed_labels))}

# Construct the interface
image = gr.inputs.Image(shape=(299,299))
label = gr.outputs.Label(num_top_classes=3)

iface = gr.Interface(
    fn=breed_prediction,
    inputs=image,
    outputs=label,
    capture_session=True,
    live=True,
    verbose=True,
    title="Dogs breed prediction from picture\nwith Xception model",
    description=gradio_desc,
    article=gradio_article,
    allow_flagging=False,
    allow_screenshot=True,
    server_port=INITIAL_PORT_VALUE,
    server_name=LOCALHOST_NAME
)

if __name__ == "__main__":
    print("server_name:", LOCALHOST_NAME)
    print("server_port:", INITIAL_PORT_VALUE)

    # iface.launch(inbrowser=True)
    iface.launch()