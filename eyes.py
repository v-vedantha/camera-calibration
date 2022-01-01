import sys
import tensorflow as tf
import tensorflow.lite as tflite
import mediapipe as mp
mp_iris = mp.solutions.iris
import cv2


class FrameProcesser:
    def __init__(self):
        #self.face_detector = mp.CalculatorGraph(graph_config=open(
        #    'mediapipe/mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt').read())
        #self.face_detector.observe_output_stream('out_stream', lambda stream_name, data: self.face_cropper.add_packet_to_input_stream('in_stream', data))
        #self.face_cropper = mp.CalculatorGraph(graph_config=open(
        #    'mediapipe/mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt').read())
        #self.face_cropper.observe_output_stream('out_stream', lambda stream_name, data: self.landmark_detector.add_packet_to_input_stream('in_stream', data))
        #self.landmark_detector = mp.CalculatorGraph(graph_config=open(
        #    'mediapipe/mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt').read())
        
        self.iris = mp_iris.Iris()

    def process(self, rgb_frame):
        results = self.iris.process(rgb_frame)

        return results
