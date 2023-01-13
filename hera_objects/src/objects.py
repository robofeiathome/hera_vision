#!/usr/bin/env python3

# Author: Brubru

import rospy
import math
import tf

from hera_objects.msg import DetectedObjectArray, ObjectPosition
from hera_objects.srv import FindObject, FindSpecificObject

class Objects:

    def __init__(self):

        self._objects = list()
        self._specific = dict()
        self._positions = dict()
        self._obj = None
        rospy.Subscriber('/dodo_detector_ros/detected', DetectedObjectArray, self.get_detected_objects)
    
        rospy.Service('objects', FindObject, self.handler)
        rospy.Service('specific_object', FindSpecificObject, self.specific_handler)

        self.listener = tf.TransformListener()
        self.reference_frame = '/manip_base_link'
        #self.reference_frame = '/map'

        self._coordinates = ObjectPosition()

        rospy.loginfo("[Objects] Dear Operator, I'm ready to give you object coordinates")

    def get_detected_objects(self, array):
        if not len(array.detected_objects) == 0:
            self._objects.clear()
            for detection in array.detected_objects:
                if detection.tf_id.data != '':
                    self._objects.append((detection.type.data, detection.tf_id.data)) # adiciona um novo objeto a lista de objetos
                else:
                    continue
            print(self._objectsbb)

    def get_positions(self, target = ''):
        self._positions.clear()
        self._specific = {0: [0.0, 0.0, 0.0]}
        for obj_class, obj_frame in self._objects: # para cada objeto da lista de objetos
            if not obj_frame == '': # se o frame do objeto não for vazio
                try: # tenta obter a posição do objeto
                    if target == '': # se não foi passado um tipo de objeto
                        trans, a = self.listener.lookupTransform(self.reference_frame, obj_frame, rospy.Time(0))
                        self._positions[obj_frame] = trans
                    
                    elif obj_class == target:
                        trans, a = self.listener.lookupTransform(self.reference_frame, obj_frame, rospy.Time(0))
                        self._positions[obj_frame] = trans
                        self._specific[0] = trans

                except Exception as e:
                    rospy.loginfo("[Objects] vish!")
                    print(e)
                    self._specific = {0: [0.0, 0.0, 0.0]}
            else:
                # retorna as posicoes zeradas
                rospy.loginfo("Posicao zerada")
                self._objects.clear()
                self._positions[obj_frame] = [0.0, 0.0, 0.0]
                self._obj = None
                self._specific = {0: [0.0, 0.0, 0.0]}

    def handler(self, request):
        condition = request.condition.lower()
        succeeded = False

        self._coordinates.x = 0.0
        self._coordinates.y = 0.0
        self._coordinates.z = 0.0
        self._coordinates.rx = 0.0
        self._coordinates.ry = 0.0
        self._coordinates.rz = 0.0
        self.get_positions()

        if condition == 'closest':
            dist = float("inf")
            for obj_id in self._positions:
                x, y, z = self._positions[obj_id]
                value = math.sqrt(x**2 + y**2 + z**2)
                if value < dist:
                    dist = value
                    self._obj = obj_id

        elif condition == 'farthest':
            dist = 0.0
            for obj_id in self._positions:
                x, y, z = self._positions[obj_id]
                value = math.sqrt(x**2 + y**2 + z**2)
                if value > dist:
                    dist = value
                    self._obj = obj_id

        elif condition == 'rightmost':
            rightmost = float('inf')
            for obj_id in self._positions:
                x, y, z = self._positions[obj_id]
                if y < rightmost:
                    rightmost = y
                    self._obj = obj_id

        elif condition == 'leftmost':
            leftmost = -float('inf')
            for obj_id in self._positions:
                x, y, z = self._positions[obj_id]
                if y > leftmost:
                    leftmost = y
                    self._obj = obj_id

        elif condition == 'higher':
            higher = -float('inf')
            for obj_id in self._positions:
                x, y, z = self._positions[obj_id]
                if z > higher:
                    higher = z
                    self._obj = obj_id

        elif condition == 'lower':
            lower = float('inf')
            for obj_id in self._positions:
                x, y, z = self._positions[obj_id]
                if z < lower:
                    lower = z
                    self._obj = obj_id

        if self._obj is not None and self._obj in self._positions:
            x, y, z = self._positions[self._obj]
            self._coordinates.x = x
            self._coordinates.y = y 
            self._coordinates.z = z
            self._coordinates.rx = 0.0
            self._coordinates.ry = 0.0
            self._coordinates.rz = math.atan2(y,x)
            succeeded = True

        rospy.loginfo('Found the coordinates!') if succeeded else rospy.loginfo("I'm a shame. Sorry!")
        rospy.loginfo(self._positions) 
        rospy.loginfo(self._obj)
        return self._coordinates

    def specific_handler(self, request):
        obj_class = request.type
        succeeded = False

        self.get_positions(obj_class) 

        rospy.loginfo(self._specific[0])

        if self._specific is not None:
            x, y, z = self._specific[0]
            self._coordinates.x = x
            self._coordinates.y = y
            self._coordinates.z = z
            self._coordinates.rx = 0.0
            self._coordinates.ry = 0.0
            self._coordinates.rz = math.atan2(y,x)
            succeeded = True

        rospy.loginfo('Found the coordinates!') if succeeded else rospy.loginfo("I'm a shame. Sorry!")

        return self._coordinates

if __name__ == '__main__':
    rospy.init_node('objects', log_level=rospy.INFO)
    Objects()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
