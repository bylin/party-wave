# -*- coding: UTF-8 -*-
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
import cv2
from tempfile import NamedTemporaryFile
app = Flask(__name__)
STREAMS = pd.read_csv('streams.tsv', sep = '\t')
SPOTS = [(index, row['name']) for index, row in STREAMS.iterrows()]
AREAS = [(index, area) for index, area in enumerate(STREAMS.loc[:, 'area'].unique())]
DEBUG = False
if not DEBUG:
  import keras_yolo

# Yolo parameters
net_h, net_w = 608, 608
obj_thresh, nms_thresh = 0.3, 0.2
# anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
anchors = [[81,82,  135,169,  344,319],  [10,14 , 23,27,  37,58]]
labels = ["surfer"]

# Build model and load weights
if not DEBUG:
  # yolov3 = keras_yolo.make_yolov3_model()
  # weight_reader = keras_yolo.WeightReader('surfer_4500.weights')
  # weight_reader.load_weights(yolov3)
  # yolov3._make_predict_function()
  surfer_cnn = keras_yolo.make_surfer_model()
  weight_reader = keras_yolo.WeightReaderTiny('surfer-tiny_8000.weights')
  weight_reader.load_weights(surfer_cnn)
  surfer_cnn._make_predict_function()

class SpotForm(Form):
  spot_id = SelectField('Spot', choices = SPOTS)
  area_id = SelectField('Area', choices = AREAS)


@app.route("/", methods=['GET'])
def index():
  form = SpotForm()
  return render_template('index.html', form=form)

@app.route("/spot", methods=['GET', 'POST'])
def spot():
  if request.method == 'GET':
    return index()
  form = SpotForm()
  spot_id = int(request.form['spot_id'])
  print('spot id: {}'.format(spot_id))
  if spot_id == -1: 
    return render_template('index.html', form=form)
  spot = probe_spot(spot_id, draw_video = True)
  return render_template('index.html', form=form, spot=spot)

@app.route("/area", methods=['GET', 'POST'])
def area():
  if request.method == 'GET':
    return index()
  form = SpotForm()
  area_id = int(request.form['area_id'])
  spot_ids = np.where(STREAMS['area'] == AREAS[area_id][1])[0]
  area = []
  for spot_id in spot_ids:
    spot = probe_spot(spot_id)
    area.append(spot)
  return render_template('index.html', form=form, area=area)

@app.route("/demo", methods=['GET'])
def demo():
  form = SpotForm()
  return render_template('demo.html', form=form)

def probe_spot(spot_id, draw_video = False):
  spot = {}
  spot['name'] = STREAMS.name[spot_id]
  spot['location'] = STREAMS.location[spot_id]
  spot['report_url'] = STREAMS.report_url[spot_id]
  spot['height'], spot['tide'] = pull_weather(spot['report_url'])
  
  stream_url = STREAMS.stream_url[spot_id]
  try:
    if not DEBUG:
      spot['crowd'] = estimate_crowds(stream_url)
    else:
      spot['crowd'] = 'Debug'

    if draw_video:
      spot['video'] = yolo_full_fps(stream_url)
  except Exception as e:
    if str(e) == 'ffmpeg error (see stderr output for detail)':
      spot['crowd'] = "N/A"
      spot['video'] = "N/A"

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
      tides_dict = json_data['spot']['report']['data']['forecast']['tide']
      tide_current = tides_dict['current']['height']
      tide1, tide2 = tides_dict['previous']['height'], tides_dict['next']['height']
      print(tides_dict)
      if tide_current > (tide1 + tide2) / 2:
        tide_current = '{:.1f} ft ⬆'.format(tide_current)
      elif tide_current == (tide1 + tide2) / 2:
        pass
      else:
        tide_current = '{:.1f} ft ⬇'.format(tide_current)
      break
  return wave_height, tide_current

def stream_to_frame_tensor(stream_url, full_fps = False):
  probe = ffmpeg.probe(stream_url)
  video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
  width = int(video_stream['width'])
  height = int(video_stream['height'])
  
  if not full_fps:
    out, err = (
      ffmpeg
      .input(stream_url, t=5)
      .filter('fps', fps=1, round='up')
      .output('pipe:', format='rawvideo', pix_fmt='rgb24')
      .run(capture_stdout=True)
    )
  else:
    out, err = (
      ffmpeg
      .input(stream_url, t=3)
      .output('pipe:', format='rawvideo', pix_fmt='rgb24')
      .run(capture_stdout=True)
    )
  video = (
    np
    .frombuffer(out, np.uint8)
    .reshape([-1, height, width, 3])
  )
  return video

def estimate_crowds(stream_url):
  frames = stream_to_frame_tensor(stream_url)
  # count boxes for each captured frame
  object_counts = []

  for i, frame in enumerate(frames):
    print('processing frame {}'.format(i))
    new_image = keras_yolo.preprocess_input(frame, net_h, net_w)
    yolos = surfer_cnn.predict(new_image)
    boxes = keras_yolo.decode_netout(yolos[1][0], anchors[1], obj_thresh, nms_thresh, net_h, net_w)
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


def yolo_full_fps(stream_url):
  probe = ffmpeg.probe(stream_url)
  video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
  width = int(video_stream['width'])
  height = int(video_stream['height'])
  process1 = (
      ffmpeg
      .input(stream_url, t=3)
      .output('pipe:', format='rawvideo', pix_fmt='rgb24')
      .run_async(pipe_stdout=True)
  )

  outfile = 'static/{:%H%M%S-%m%d%y}.mp4'.format(datetime.now())
  process2 = (
      ffmpeg
      .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
      .output(outfile, pix_fmt='yuv420p')
      .overwrite_output()
      .run_async(pipe_stdin=True)
  )
  
  while True:
    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes:
        break
    frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([height, width, 3])
    )

    new_image = keras_yolo.preprocess_input(frame, net_h, net_w)
    yolos = surfer_cnn.predict(new_image)
    boxes = keras_yolo.decode_netout(yolos[1][0], anchors[1], obj_thresh, nms_thresh, net_h, net_w)
    image_h, image_w, _ = frame.shape
    keras_yolo.correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    keras_yolo.do_nms(boxes, nms_thresh)
    keras_yolo.draw_boxes(frame, boxes, labels, obj_thresh)    

    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )

  process2.stdin.close()
  process1.wait()
  process2.wait()

  return outfile
