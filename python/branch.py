import numpy as np
import cv2

def get_circle(r):
    """
    주어진 반지름을 이용하여 원을 생성하는 함수입니다.

    Args:
    - r (int): 생성하고자 하는 원의 반지름

    Returns:
    - circle (numpy.ndarray): 생성된 원
    """
    circle = np.zeros((2*r+1, 2*r+1))
    for i in range(2*r+1):
        for j in range(2*r+1):
            if (i-r)**2 + (j-r)**2 <= r**2:
                circle[i,j] = 1
    return circle

if __name__ == "__main__":
    print(get_circle(6))