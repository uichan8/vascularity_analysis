# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

from mahotas.morph import hitmiss as hit_or_miss

# find center point from skeleton------------------------
class bifur_point:
    def __init__(self):
        self.X = self.X_list()
        self.T = self.T_list()
        self.Y = self.Y_list()

    @staticmethod
    def X_list():
        X0 = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
        X1 = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [1, 0, 1]])

        return [X0,X1]

    @staticmethod
    def T_list():
        T0=np.array([[2, 1, 2],
                    [1, 1, 1],
                    [2, 2, 2]])

        T1=np.array([[1, 2, 1],
                    [2, 1, 2],
                    [1, 2, 2]])

        T2=np.array([[2, 1, 2],
                    [1, 1, 2],
                    [2, 1, 2]])

        T3=np.array([[1, 2, 2],
                    [2, 1, 2],
                    [1, 2, 1]])

        T4=np.array([[2, 2, 2],
                    [1, 1, 1],
                    [2, 1, 2]])

        T5=np.array([[2, 2, 1],
                    [2, 1, 2],
                    [1, 2, 1]])

        T6=np.array([[2, 1, 2],
                    [2, 1, 1],
                    [2, 1, 2]])

        T7=np.array([[1, 2, 1],
                    [2, 1, 2],
                    [2, 2, 1]])
        return[T0,T1,T2,T3,T4,T5,T6,T7]

    @staticmethod
    def Y_list():
        Y0=np.array([[1, 0, 1],
                    [0, 1, 0],
                    [2, 1, 2]])

        Y1=np.array([[0, 1, 0],
                    [1, 1, 2],
                    [0, 2, 1]])

        Y2=np.array([[1, 0, 2],
                    [0, 1, 1],
                    [1, 0, 2]])

        Y2=np.array([[1, 0, 2],
                    [0, 1, 1],
                    [1, 0, 2]])

        Y3=np.array([[0, 2, 1],
                    [1, 1, 2],
                    [0, 1, 0]])

        Y4=np.array([[2, 1, 2],
                    [0, 1, 0],
                    [1, 0, 1]])
        Y5=np.rot90(Y3)
        Y6 = np.rot90(Y4)
        Y7 = np.rot90(Y5)
        return [Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7]

class end_point:
    def __init__(self):
        self.END = self.END_list

    def END_list(self):
        endpoint1=np.array([[0, 0, 0],
                            [0, 1, 0],
                            [2, 1, 2]])

        endpoint2=np.array([[0, 0, 0],
                            [0, 1, 2],
                            [0, 2, 1]])

        endpoint3=np.array([[0, 0, 2],
                            [0, 1, 1],
                            [0, 0, 2]])

        endpoint4=np.array([[0, 2, 1],
                            [0, 1, 2],
                            [0, 0, 0]])

        endpoint5=np.array([[2, 1, 2],
                            [0, 1, 0],
                            [0, 0, 0]])

        endpoint6=np.array([[1, 2, 0],
                            [2, 1, 0],
                            [0, 0, 0]])

        endpoint7=np.array([[2, 0, 0],
                            [1, 1, 0],
                            [2, 0, 0]])

        endpoint8=np.array([[0, 0, 0],
                            [2, 1, 0],
                            [1, 2, 0]])

        return [endpoint1,endpoint2,endpoint3,endpoint4,endpoint5,endpoint6,endpoint7,endpoint8]

B_POINT = bifur_point()
E_POINT = end_point()

def find_bifur_points(skel):
    bp = np.zeros(skel.shape, dtype=int)
    for x in B_POINT.X:
        bp = bp + hit_or_miss(skel,x)
    for y in B_POINT.Y:
        bp = bp + hit_or_miss(skel,y)
    for t in B_POINT.T:
        bp = bp + hit_or_miss(skel,t)
    return bp

def find_end_points(skel):
    ep = 0
    for e in END_POINT.END:
        ep = ep + hit_or_miss(skel,e)
    return ep

#--------------------------------------------------------

#find center mask----------------------------------------

class circle:
    def __init__(self,max_r = 5):
        self._circle_edge_list = []
        self._circle_mask_list = []

        for i in range(max_r+1):
            if i % 2 == 1:
                self._circle_edge_list.append(self.get_circle(i))
                self._circle_mask_list.append(self.get_circle_mask(i))

    @staticmethod
    def angle(x_center, y_center, x, y):
        dx = x - x_center
        dy = y - y_center
        radians = math.atan2(dy, dx)
        degrees = math.degrees(radians)
        return degrees if degrees >= 0 else 360 + degrees

    def get_circle(self,r):
        x = 0
        y = r
        d = 3 - (2 * r)
        pixels = set()

        while x <= y:
            pixels.add((x, y))
            pixels.add((y, x))
            pixels.add((x, y))
            pixels.add((y, x))
            pixels.add((x, y))
            pixels.add((y, x))
            pixels.add((x, y))
            pixels.add((y, x))

            if d < 0:
                d = d + (4 * x) + 6
            else:
                d = d + 4 * (x - y) + 10
                y -= 1
            x += 1

        for pixel in list(pixels):
            pixels.add((-pixel[0],pixel[1]))
            pixels.add((pixel[0],-pixel[1]))
            pixels.add((-pixel[0],-pixel[1]))

        circle_coor = sorted(pixels, key=lambda p: self.angle(0, 0, p[0], p[1]))
        return np.array(circle_coor)

    @staticmethod
    def get_circle_mask(radius):
        mask_shape = 2 * radius + 1
        mask = np.zeros([mask_shape, mask_shape])
        for y in range(mask_shape):
            for x in range(mask_shape):
                if (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2:
                    mask[y, x] = 1
        return mask

    @property
    def circle_edge_list(self):
        return self._circle_edge_list
    
    @property
    def circle_mask_list(self):
        return self._circle_mask_list
    
C = circle()

def get_pixel_values(mask, coordinates): 
    pixel_values = []
    for x, y in coordinates:
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            pixel_values.append(mask[y, x])
        else:
            pixel_values.append(0)
    pixel_values.append(pixel_values[0])
    return np.array(pixel_values)

def find_bifur_mask(mask,x,y,max_r = 5):
    dilated_mask = cv2.dilate(mask, np.ones((3,3), dtype=np.uint8), iterations=1)
    i = 0
    for circle_edge in C.circle_edge_list:
        r = 2 * i + 1
        coor = circle_edge + np.array([x,y])
        l = get_pixel_values(mask, coor)
        if (l[1:] != l[:-1]).sum() > 5 or r == max_r:
            break
        i += 1
        
    bifur_mask = mask[y-r:y+r+1,x-r:x+r+1] * C.circle_mask_list[i]
    return np.array(bifur_mask,dtype=np.uint8)
    
#--------------------------------------------------------