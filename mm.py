from tflite_runtime.interpreter import Interpreter
import time
from PIL import Image
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import RPi.GPIO as GPIO    # Import Raspberry Pi GPIO library
from time import sleep

GPIO.setwarnings(False)    # Ignore warning for now
GPIO.setmode(GPIO.BOARD)   # Use physical pin numbering
pins = [12,16,18]
for pin in pins:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

def get_output_led(label_id):
    GPIO.setup(pins[label_id], GPIO.HIGH)
    sleep(10)
    GPIO.setup(pins[label_id], GPIO.LOW)
    return

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))
  print(output)

  return [np.argmax(output), output[np.argmax(output)]*100]

#initialize camera
camera = PiCamera()
camera.resolution = (256, 256)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(256, 256))

#allow camera to warm up
time.sleep(0.1)

#capture image
image = np.empty((256, 256, 3), dtype=np.uint8)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

#convert image to pillow image
image = Image.fromarray(image)

tflite=Interpreter('/home/devin/waste_classify/model.tflite')
tflite.allocate_tensors()
input_details=tflite.get_input_details()
output_details=tflite.get_output_details()

curr = time.time()

classes = ['organic', 'recyclable', 'trash']

path = '/home/devin/waste_classify/download.jpeg'

_, height, width, _ = tflite.get_input_details()[0]['shape']
print("Image Shape (", width, ",", height, ")")

# Load an image to be classified.
#image = Image.open(path).convert('RGB').resize((width, height))

# Classify the image.
time1 = time.time()
label_id, prob = classify_image(tflite, image)
time2 = time.time()
classification_time = np.round(time2-time1, 3)
print("Classificaiton Time =", classification_time, "seconds.")

# Read class labels.
labels = classes

# Return the classification label of the image.
classification_label = labels[label_id]

print("Image Label is :", classification_label, ", with Accuracy :", prob, "%.")

get_output_led(label_id)
print ("nisha " nisha rjapati gaet label)
