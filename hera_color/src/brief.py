#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import matplotlib.pyplot as plt


class LoadFeature(object):

    def __init__(self):
    
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.camera_callback)
        self.bridge_object = CvBridge()
        self.x = 1000

    def camera_callback(self,data):
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        

        image_1 = cv2.imread('/home/hera/catkin_hera/src/3rdParty/vision_system/hera_face/images/refri.png',1) # caminho da imagem para ser detectada
        image_2 = cv_image

        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

        # Inicializar o detector de recurso ORB
        orb = cv2.ORB_create(nfeatures = 100000)

        # Faça uma cópia da imagem original para exibir os pontos-chave encontrados pelo ORB
        # Este é apenas um representante
        preview_1 = np.copy(image_1)
        preview_2 = np.copy(image_2)

        # Crie outra cópia para exibir apenas pontos
        dots = np.copy(image_1)

        # Extraia os pontos-chave de ambas as imagens
        train_keypoints, train_descriptor = orb.detectAndCompute(gray_1, None)
        test_keypoints, test_descriptor = orb.detectAndCompute(gray_2, None)

        # Desenhe os pontos-chave encontrados na imagem principal
        cv2.drawKeypoints(image_1, train_keypoints, preview_1, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.drawKeypoints(image_1, train_keypoints, dots, flags=2)

        #############################################
        ################## MATCHER ##################
        #############################################

        # Inicializar o BruteForce Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        # Corresponda aos pontos característicos de ambas as imagens
        matches = bf.match(train_descriptor, test_descriptor)

        # As partidas com distâncias mais curtas são as que queremos.
        matches = sorted(matches, key = lambda x : x.distance)
        # Pegue alguns dos pontos correspondentes para desenhar
        
            
        good_matches = matches[:300] # ESTE VALOR FOI ALTERADO VOCÊ VERÁ MAIS TARDE PORQUE
        

        # Passe os pontos de recurso
        train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Crie uma máscara para capturar os pontos correspondentes
        # Com a homografia estamos tentando encontrar perspectivas entre dois planos
        # Usando o método RANSAC não determinístico
        M, mask = cv2.findHomography(train_points, test_points, cv2.RANSAC,5.0)

        # Capture a largura e a altura da imagem principal
        h,w = gray_1.shape[:2]

        # Crie uma matriz flutuante para a nova perspectiva
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Crie a perspectiva no resultado
        dst = cv2.perspectiveTransform(pts,M)

        # Desenhe as linhas correspondentes
        

        # Desenhe os pontos da nova perspectiva na imagem resultante (é considerada a caixa delimitadora)
        result = cv2.polylines(image_2, [np.int32(dst)], True, (50,0,255),3, cv2.LINE_AA)


        cv2.imshow('Points',preview_1)

        cv2.imshow('Dots',dots)
        
        cv2.imshow('Detection',image_2)       

        cv2.waitKey(1)

def main():
    load_feature_object = LoadFeature()
    rospy.init_node('load_feature_node', anonymous=True)
    
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
