## Some examples of how ML model can be integrated with the backend of a Flutter App:
1: Tensorflow Lite with Teachable Machine
Once we are satisfied with the accuracy of the model, we can export the model and convert it to the right format based on your platform(TensorFlow Lite).
In this case, we can export the custom model as Tensorflow Lite model for mobile apps.

Once we've exported the model as a downloadable zip file, we will receive 2 files — labels.txt and model_unquant.tflite.
The labels.txt file is a list of data labels for the model to classify data inputs accordingly.
The model_unquant.tflite file is the actual custom model in Tensforflow Lite format.

we can either embed these two files in the 'assets' folder of the Flutter project or we can upload the model to the Firebase ML for serving dynamically to the mobile apps.

![](App/assets.jpg)
