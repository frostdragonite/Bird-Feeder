import os
import threading
import time
import humanfriendly

import cv2
import libcamera
import numpy as np
import pystream
import tensorflow as tf
from gpiozero import MotionSensor
from picamera2 import Picamera2, Preview
from tensorflow.keras.models import load_model

import load_detector_batch as detector_batch

from telegram.ext import Updater

# Set up the PIR sensor and camera
pir = MotionSensor(4)
camera = Picamera2()
camera_config = camera.create_video_configuration(main={"size": (640, 480)})
camera_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
camera.configure(camera_config)

# Set up Machine Learning Model
model_MD = 'md_v5a.0.0.pt'
model_Bird1 = os.path.join('models','bird1.h5')
model_Bird2 = os.path.join('models','birdLB1.h5')

TELEGRAM_BOT_TOKEN = "Your Telegram Bot Token"

# # Set up the video stream
# client_socket = socket.socket()
# client_socket.connect(('localhost', 8000))
# connection = client_socket.makefile('wb')
# stream = io.BytesIO()

# # Avoid OOM errors by setting GPU Memory Consumption Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

# Function: Create name for captured images
def get_image_name():
  path, dirs, files = next(os.walk('Image'))
  file_count = len(files)
  return 'Image/' + str(file_count + 1) + '.jpg'

# Function: Capturing picture and save it
def take_picture():
  image_name = get_image_name()
  print("Captured!")
  camera.capture_file(image_name)
  return image_name

# Function: Processing image with MegaDetector model
def process_image(file_name):
  return detector_batch.process_image(file_name, model_1, 
                                      confidence_threshold=0.6, quiet=False, 
                                      image_size=None)
  
# Function: Check if there are animals, return boundary boxes for every detected animals
def is_animal_image(ml_result):
  result = []
  for detection in ml_result['detections']:
    if detection['category'] == '1' and detection['conf'] >= 0.7:
      result.append(detection['bbox'])
  
  return result

# Function: Crop image for classifying
def crop_image(file_name, boundary_box):
  img = cv2.imread(file_name)
  height = img.shape[0]
  width = img.shape[1]
  x=round(boundary_box[0] * width)
  y=round(boundary_box[1] * height)
  w=round(x+boundary_box[2] * width)
  h=round(y+boundary_box[3] * height)
  # print(x,y,w,h)
  return img[y:h, x:w]
  
# Function: Classifier model
def check_bird_in_pics(image_path, animal_boxes):
  # Crop each animals and predict
  bird = []
  for box in animal_boxes:
    img = crop_image(image_path, box)
    resize = tf.image.resize(img, (256,256))
    
    # Check if bird or not bird
    score = model_2.predict(np.expand_dims(resize/255, 0))
    if score > 0.5:
      print(f'This one is not bird')
    else:
      
      # Check if bird or like bird
      score = model_3.predict(np.expand_dims(resize/255, 0))
      if score > 0.5:
        print(f'This one is not bird')
      else:
        bird_image_name = os.path.join('BirdImage',image_path.split('/')[-1][:-4] + '_bird' + str(len(bird)) + '.jpg')
        cv2.imwrite(bird_image_name, img)
        bird.append(bird_image_name)
        print(f'This one is bird')
        
  # All checks
  return bird

# Function: Send message in telegram
def send_to_telegram(filepath):
  for attempt in range(10):
    try:
      print("Sending to telegram {0}".format(filepath))
      with open(filepath, "rb") as imagefile:
        updater.bot.send_photo(
              chat_id="2084600136",
              photo=imagefile,
              caption="Hello! I notice a bird! Check the camera! >> http://192.168.1.61:8000/",
        )
        break
                # with open(filepath, "rb") as videofile:
                #     updater.bot.send_video(
                #         chat_id="2084600136",
                #         video=videofile,
                #         duration=60,
                #         caption="Motion detected!",
                #         supports_streaming=True,
                #         disable_notification=True,
                #         timout=30,
                #     )
                #     print("{0} sent successfully ".format(filepath))
                #     break
    except Exception as e:
      print(e)
      print('Module "%s": tg connection error {0}, retry.'.format(e))
      time.sleep(5)

  return None

# Function: Send cropped pic
def send_bird_telegram(filepath):
  for attempt in range(10):
    try:
      with open(filepath, "rb") as imagefile:
        updater.bot.send_photo(
              chat_id="2084600136",
              photo=imagefile,
              caption="and here is a bird from it.",
        )
        break
                # with open(filepath, "rb") as videofile:
                #     updater.bot.send_video(
                #         chat_id="2084600136",
                #         video=videofile,
                #         duration=60,
                #         caption="Motion detected!",
                #         supports_streaming=True,
                #         disable_notification=True,
                #         timout=30,
                #     )
                #     print("{0} sent successfully ".format(filepath))
                #     break
    except Exception as e:
      print(e)
      print('Module "%s": tg connection error {0}, retry.'.format(e))
      time.sleep(5)

  return None

# Thread: Running the streaming server
def streaming_server():
  pystream.start_streaming(camera)


# Thread: Capturing images and process
def image_capture():
  # Loop over detections from the PIR sensor
  while True:
    
    # Wait for motion detected
    pir.wait_for_motion()
    print("You moved, processing")
    time.sleep(3)
    
    # Take a picture and save it to disk
    image_name = take_picture()
    
    # Processing image
    start_time = time.time()
    ml_result = process_image(image_name)
    elapsed = time.time() - start_time
    print('Done processing in {}'.format(humanfriendly.format_timespan(elapsed)))
    results = is_animal_image(ml_result)
    if results:
      print('Animal detected!', len(results), results)
      
      # Check if there is a bird in the picture
      birbs = check_bird_in_pics(image_name,results)
      if birbs:
        print('Done! and there is bird!!! :)')
        send_to_telegram(image_name)
        for bird in birbs:
          send_bird_telegram(bird)
      else:
        print('Done! and.. those are not bird :(')
        
    else:
      print('No animal detected')


    # Snooze for a bit
    pir.wait_for_no_motion()
    print("You out")
    time.sleep(5)
    
# Load the model
model_1 = detector_batch.detector_batch_loader(model_MD)
model_2 = load_model(model_Bird1)  
model_3 = load_model(model_Bird2)
print("Done loading all the model")

# Setup Telegram
updater = Updater(TELEGRAM_BOT_TOKEN, use_context = True)

# Create two threads
server_thread = threading.Thread(target=streaming_server)
capture_thread = threading.Thread(target=image_capture)

# Start both threads
server_thread.start()
capture_thread.start()

# Wait for both threads to finish
server_thread.join()
capture_thread.join()