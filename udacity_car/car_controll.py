# https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/drive.py
import base64
from datetime import datetime
import os

import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from udacity_car.data_loader import pre_process
import torch
from torch.autograd import Variable

sio = socketio.Server()
app = Flask(__name__)
prev_image_array = None

model = torch.load('model.pkl')


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        tensor_img = pre_process(image)

        tensor_img = torch.cat([tensor_img, tensor_img, tensor_img])
        tensor_img.unsqueeze_(0)
        tt = Variable(tensor_img)

        if torch.cuda.is_available():
            tt = tt.cuda()

        output = model(tt).cpu().data[0]

        steering_angle = output[0]
        throttle = output[1]
        brake = output[2]

        print(steering_angle, throttle, brake)

        send_control(steering_angle, throttle, brake)

        # save frame
        # timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        # image_filename = os.path.join('./img', timestamp)
        # image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 0)


def send_control(steering_angle, throttle, brake):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
            # 'brake': brake.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
