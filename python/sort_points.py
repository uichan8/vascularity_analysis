import numpy as np
import matplotlib.pyplot as plt

def find_end_point(target_line_mask ,point = None , track_path = False):
    """
    이미지 상에서 주어진 라인 마스크의 끝점을 찾는 함수입니다.

    Args:
    - target_line_mask (numpy.ndarray): 입력 이미지 상에서 추출한 라인 마스크
    - track_path (bool, optional): 경로를 추적하여 반환할 지 여부를 결정하는 bool 값. 기본값은 False입니다.

    Returns:
    - end_point (tuple): 라인 마스크의 끝점 좌표

    - (optional) new_x (list), new_y (list): 라인 마스크의 경로를 추적한 결과 좌표 리스트
    """
    if point is None:
        vertex_r = np.where(target_line_mask)
        point = (vertex_r[1][0], vertex_r[0][0])

    new_x = []
    new_y = []

    while(target_line_mask[point[1]-1:point[1]+2, point[0]-1:point[0]+2].sum() != 1):
        target_line_mask[point[1],point[0]] = 0
        new_x.append(point[0])
        new_y.append(point[1])
        neighbors =[
            (point[0] - 1, point[1]), 
            (point[0] + 1, point[1]), 
            (point[0], point[1] - 1), 
            (point[0], point[1] + 1),
            (point[0] - 1, point[1] - 1),
            (point[0] - 1, point[1] + 1),
            (point[0] + 1, point[1] - 1),
            (point[0] + 1, point[1] + 1)]
        for neighbor in neighbors:
            if target_line_mask[neighbor[1],neighbor[0]]:
                point = (neighbor[0],neighbor[1])
    new_x.append(point[0])
    new_y.append(point[1])
    if track_path:
        return new_x, new_y
    return point

def sort_points(target_line_mask):
    """
    이미지 상에서 주어진 라인 마스크와 시작점 좌표를 이용하여 라인의 경로를 추적하고, 추적된 좌표를 반환하는 함수입니다.

    Args:
    - target_line_mask (numpy.ndarray): 입력 이미지 상에서 추출한 라인 마스크
    - point (tuple): 라인의 시작점 좌표

    Returns:
    - new_x (list): 라인의 경로를 추적한 결과 x좌표 리스트
    - new_y (list): 라인의 경로를 추적한 결과 y좌표 리스트
    """
    end_point = find_end_point(target_line_mask.copy())
    new_x, new_y = find_end_point(target_line_mask.copy(), end_point, True)

    return new_x, new_y
    
