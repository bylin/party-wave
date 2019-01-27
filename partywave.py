from flask import Flask
from flask import render_template
from flask import request
from wtforms import Form, SelectField
import pandas as pd
import numpy as np
import requests
import json
import ffmpeg
from datetime import datetime
import keras_yolo
import cv2
app = Flask(__name__)
STREAMS = pd.read_csv('streams.tsv', sep = '\t')
CHOICES = [(-1, '')] + [(index, row['name']) for index, row in STREAMS.iterrows()]
DEBUG = False

# Preload yolo model and weights

# Yolo parameters
net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.3, 0.45
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]

# Build model and load weights
yolov3 = keras_yolo.make_yolov3_model()
weight_reader = keras_yolo.WeightReader('surfer_4500.weights')
labels = ["surfer"]

if not DEBUG:
  weight_reader.load_weights(yolov3)
  yolov3._make_predict_function()

class SpotForm(Form):
  spot_id = SelectField('Spot', choices = CHOICES)

@app.route("/", methods=['GET', 'POST'])
def index():
  form = SpotForm()
  if request.method == 'POST':
    spot_id = int(request.form['spot_id'])
    if spot_id == -1: 
      return render_template('index.html', form=form)
    
    spot = probe_spot(spot_id)
    spots = [spot]
    return render_template('index.html', form=form, spots=spots)
  
  return render_template('index.html', form=form)

def probe_spot(spot_id):
  spot = {}
  spot['name'] = STREAMS.name[spot_id]
  spot['location'] = STREAMS.location[spot_id]
  spot['report_url'] = STREAMS.report_url[spot_id]
  spot['height'], spot['tide'] = pull_weather(spot['report_url'])
  
  stream_url = STREAMS.stream_url[spot_id]
  if not DEBUG:
    spot['crowd'] = estimate_crowds(stream_url)
  else:
    spot['crowd'] = 'Debug'

  return spot

def pull_weather(report_url):
  r = requests.get(report_url)
  json_data = ''
  swells = []
  for i, line in enumerate(r.text.split('\n')):
    if line[:21] == '      window.__DATA__':
      json_string = line[24:]
      json_data = json.loads(json_string)
      json_data['spot']['report']['data']
      wave_height_dict = json_data['spot']['report']['data']['forecast']['waveHeight']
      wave_height = '{} - {} ft'.format(wave_height_dict['min'], wave_height_dict['max'])
      tide = json_data['spot']['report']['data']['forecast']['tide']['current']['type'].title()
      break
  return wave_height, tide

def estimate_crowds(stream_url):
  frames = stream_to_frame_tensor(stream_url)
  # count boxes for each captured frame
  object_counts = []

  for i, frame in enumerate(frames):
    print('processing frame {}'.format(i))
    new_image = keras_yolo.preprocess_input(frame, net_h, net_w)
    yolos = yolov3.predict(new_image)
    boxes = keras_yolo.decode_netout(yolos[2][0], anchors[2], obj_thresh, nms_thresh, net_h, net_w)
    image_h, image_w, _ = frame.shape
    keras_yolo.correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    keras_yolo.do_nms(boxes, nms_thresh)    

    object_counts.append(sum([box.classes[0] > obj_thresh for box in boxes]))

  nsurfers = np.max(object_counts)
  if nsurfers > 20:
    return 'Gnarly'
  elif nsurfers > 10:
    return '🎊 Party 🎉'
  return 'Sparse'

def stream_to_frame_tensor(stream_url):
    probe = ffmpeg.probe(stream_url)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    
    out, err = (
      ffmpeg
      .input(stream_url, t=5)
      .filter('fps', fps=1, round='up')
      .output('pipe:', format='rawvideo', pix_fmt='rgb24')
      .run(capture_stdout=True)
    )
    video = (
      np
      .frombuffer(out, np.uint8)
      .reshape([-1, height, width, 3])
    )
    return video



