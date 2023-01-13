#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf
import pandas
import cv2
import matplotlib.pyplot as plt 
import torch

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2

from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort import generate_detections
from deep_sort import preprocessing as prep

from tracker.msg import TrackerObject, TrackerObjectArray



class Detector: 

   def __init__(self):
      self.deep_weights = '/home/hera/catkin_hera/src/3rdParty/vision_system/tracker/src/deep_sort/model_data/mars-small128.pb'
      self.yolo_weights = '/home/hera/catkin_hera/src/3rdParty/vision_system/tracker/src/yolov5/weights/yolov5s.pt'
      self.yolov5 = '/home/hera/catkin_hera/src/3rdParty/vision_system/tracker/src/yolov5'
      image_topic = rospy.get_param('~image_topic')
      point_cloud_topic = rospy.get_param('~point_cloud_topic', None)

      self._global_frame = 'camera'
      self._frame = 'camera_depth_frame'
      self._tf_prefix = rospy.get_param('~tf_prefix', rospy.get_name())

      global detect
      global scores
      global tracker
      global encoder


      self.id = 0
      # create a transform listener so we get the fixed frame the user wants
      # to publish object tfs related to
      self._tf_listener = tf.TransformListener()


      # create detector
      self._bridge = CvBridge()

      # image and point cloud subscribers
      # and variables that will hold their values
      rospy.Subscriber(image_topic, Image, self.image_callback)

      if point_cloud_topic is not None:
         rospy.Subscriber(point_cloud_topic, PointCloud2, self.pc_callback)
      else:
         rospy.loginfo(
            'No point cloud information available. Objects will not be placed in the scene.')

      self._current_image = None
      self._current_pc = None

      # publisher for frames with detected objects
      self._imagepub = rospy.Publisher('~labeled_detect', Image, queue_size=10)

      self._tfpub = tf.TransformBroadcaster()
      rospy.loginfo('Ready to detect!')

   def image_callback(self, image):
      """Image callback"""
      # Store value on a private attribute
      self._current_image = image

   def pc_callback(self, pc):
      """Point cloud callback"""
      # Store value on a private attribute
      self._current_pc = pc

   def run(self):
      # run while ROS runs
      while not rospy.is_shutdown():
         # only run if there's an image present
         if self._current_image is not None:
            try:

                # if the user passes a fixed frame, we'll ask for transformation
                # vectors from the camera link to the fixed frame
                if self._global_frame is not None:
                  (trans, _) = self._tf_listener.lookupTransform('/' + self._global_frame,
                                                                 '/'+ self._frame,
                                                                 rospy.Time(0))

                # convert image from the subscriber into an OpenCV image
                scene = self._bridge.imgmsg_to_cv2(self._current_image, 'rgb8')

                #################################################
                model = torch.hub.load(self.yolov5, 'custom', path=self.yolo_weights, source='local')
                model.classes = [0]
                model.eval()
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.to(device)

                frame = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
                # Inference
                results = model(scene, size=640)
                width = scene.shape[1]
                height = scene.shape[0]
                results.xyxy[0] 
                deteccao = results.pandas().xyxy[0]

                pred = np.array(deteccao[['xmin', 'ymin', 'xmax', 'ymax']])
                
                marked = results.render()
                marked_image = np.squeeze(marked)
                self._imagepub.publish(self._bridge.cv2_to_imgmsg(marked_image, 'rgb8'))

                bboxes = []
                scores = []

                for i in range(len(deteccao)):
                    # (x, y, width, height)
                    #detect.append(np.array([int(deteccao['xmin'][i]), int(deteccao['ymin'][i]), width, height]))
                    scores.append(float('%.2f' % deteccao['confidence'][i]))
                    bbox_left = int(deteccao['xmin'][i])
                    bbox_top = int(deteccao['ymin'][i])
                    bbox_w = 0 #abs(int(deteccao['xmax'][i]) - int(deteccao['xmin'][i]))
                    bbox_h = abs(int(deteccao['ymax'][i]) - int(deteccao['ymin'][i]))
                    box = np.array([bbox_left, bbox_top, bbox_w, bbox_h])
                    bboxes.append(box)
                bboxes = np.array(bboxes)
                scores = np.array(scores)
                print(type(bboxes))
                #detect = np.array(detect) 
                metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 300)
                tracker = Tracker(metric, max_iou_distance = 0.7, max_age = 70, n_init = 3)
                model_filename = self.deep_weights #Change it to your directory
                encoder = generate_detections.create_box_encoder(model_filename)

                features = encoder(frame, bboxes) 
                detections_new = [Detection(bbox, score, feature) for bbox,score, feature in zip(bboxes,scores, features)] # cria um objeto detection para cada deteccao
                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections_new]) # cria um array com as coordenadas de cada deteccao
                print(type(boxes))
                scores_new = np.array([d.confidence for d in detections_new]) # cria um array com as confiancas de cada deteccao
                indices = prep.non_max_suppression(boxes, 1.0 , scores_new) # aplica o non-maxima suppression para eliminar deteccoes duplicadas
                detections_ob = [detections_new[i] for i in indices] # cria um array com as deteccoes que sobraram
                # Call the tracker
                tracker.predict() # faz a predicao do tracker
                tracker.update(detections_ob) # atualiza o estado do objeto

                publishers = {}

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1: # se o objeto foi confirmado e nao foi atualizado a mais de 1 frame
                        print("Objeto nao confirmado ou nao atualizado")
                        continue
                    bbox = track.to_tlbr()
                    print("Nao entrou")
                    ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
                    id = track.track_id
                    publishers[id] = rospy.Publisher('~person_' + str(id), TrackerObjectArray, queue_size=10)
                    #mark = marked_image[int(ymin):int(ymax), int(xmin):int(xmax)]

                    #print(bbox.shape)
                    #print(bbox)
                    #self._imagepub.publish(self._bridge.cv2_to_imgmsg(bbox, 'rgb8'))
                    #print(bbox)
                    #self._imagepub.publish(self._bridge.cv2_to_imgmsg(bbox, 'rgb8'))
                    publish_tf = False
                    if self._current_pc is None:
                        rospy.loginfo(
                            'No point cloud information available to track current object in scene')

                    # if there is point cloud data, we'll try to place a tf
                    # in the object's location
                    else:
                        y_center = round(ymax - ((ymax - ymin) / 2))
                        x_center = round(xmax - ((xmax - xmin) / 2))
                        # this function gives us a generator of points.
                        # we ask for a single point in the center of our object.
                        pc_list = list(
                            pc2.read_points(self._current_pc,
                                        skip_nans=True,
                                        field_names=('x', 'y', 'z'),
                                        uvs=[(x_center, y_center)]))

                        if len(pc_list) > 0:
                            publish_tf = True
                            # this is the location of our object in space
                            tf_id = 'Person' + '_' + str(id)

                            # if the user passes a tf prefix, we append it to the object tf name here
                            if self._tf_prefix is not None:
                                tf_id = self._tf_prefix + '/' + tf_id

                            tf_id = tf_id

                            point_x, point_y, point_z = pc_list[0]

                    # we'll publish a TF related to this object only once
                    if publish_tf:
                        # kinect here is mapped as camera_link
                        # object tf (x, y, z) must be
                        # passed as (z,-x,-y)
                        object_tf = [point_z, -point_x, -point_y]
                        frame = self._frame

                        # translate the tf in regard to the fixed frame
                        if self._global_frame is not None:
                            object_tf = np.array(trans) + object_tf
                            frame = self._global_frame

                        # this fixes #7 on GitHub, when applying the
                        # translation to the tf creates a vector that
                        # RViz just can'y handle
                        if object_tf is not None:
                            self._tfpub.sendTransform((object_tf),
                                                    tf.transformations.quaternion_from_euler(
                                                    0, 0, 0),
                                                    rospy.Time.now(),
                                                    tf_id,
                                                    frame)


            except CvBridgeError as e:
                print(e)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(e)


if __name__ == '__main__':
   rospy.init_node('tracker_node', log_level=rospy.INFO)

   try:
      Detector().run()
   except KeyboardInterrupt:
      rospy.loginfo('Shutting down')
