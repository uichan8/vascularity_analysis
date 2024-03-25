import numpy as np
import cv2
import matplotlib.pyplot as plt

def mask_width_detection(seg_mask, pts_arr):
    x1_edge = np.array([])
    x2_edge = np.array([])
    y1_edge = np.array([])
    y2_edge = np.array([])

    for i in range(0, len(pts_arr)-1,2):
        if i == len(pts_arr)-1:
            (x, y) = (pts_arr[i][0], pts_arr[i][1])
            (diff_x, diff_y) = (pts_arr[i][0] - pts_arr[i-1][0], pts_arr[i][1] - pts_arr[i-1][1])
        else:       
            (x, y) = (pts_arr[i][0], pts_arr[i][1])
            (diff_x, diff_y) = (pts_arr[i+1][0] - pts_arr[i][0], pts_arr[i+1][1] - pts_arr[i][1])
            if diff_x == 0 and diff_y == 0:
                (diff_x, diff_y) = (pts_arr[i+2][0] - pts_arr[i][0], pts_arr[i+2][1] - pts_arr[i][1])
            (normal_x, normal_y) = (diff_y, -diff_x)
        
        x1 = x
        x2 = x
        y1 = y
        y2 = y

        while(x1<seg_mask.shape[1] and y1< seg_mask.shape[0]):
            x_prime = x1 + normal_x
            x_pprime = x2 - normal_x
            y_prime = y1 + normal_y
            y_pprime = y2 - normal_y

            # img[int(x_prime), int(y_prime)] = [255, 255, 255]
            # img[int(x_prime), int(y_pprime)] = [255, 255, 255]
            if seg_mask[int(y_prime), int(x_prime)] == 0 and seg_mask[int(y_pprime), int(x_pprime)] == 0:
                x1_edge = np.append(x1_edge,x_prime)
                x2_edge = np.append(x2_edge, x_pprime)
                y1_edge = np.append(y1_edge,y_prime)
                y2_edge = np.append(y2_edge,y_pprime)
                break
            if seg_mask[int(y_prime), int(x_prime)] == 0:
                x_prime = x1
                y_prime = y1
            if seg_mask[int(y_pprime), int(x_pprime)] == 0:
                x_pprime = x2
                y_pprime = y2

            x1 = x_prime
            x2 = x_pprime
            y1 = y_prime
            y2 = y_pprime 

    return x1_edge , x2_edge, y1_edge, y2_edge

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

def simple_sampling(arr, sparsity):
    sample_arr = []
    for i in range(len(arr)):
        if i % sparsity == 0:
            sample_arr.append(arr[i])
    arr = sample_arr

    return np.array(arr)

def delete_outliers(x_data, y_data, r_data, diff_data , threshhold = 3):
    mean = sum(r_data) / len(r_data)
    variance = sum((x - mean) ** 2 for x in r_data)/ len(r_data)
    std_dev = np.sqrt(variance)
    r_copy = r_data.copy()
    count = 0

    for i in range(len(r_copy)) :
        score = (r_copy[i] - mean) / std_dev
        if abs(score) > threshhold:
            x_data = np.delete(x_data, i-count)
            y_data = np.delete(y_data, i-count)
            r_data = np.delete(r_data, i-count)
            diff_data = np.delete(diff_data, i-count)
            count += 1

    return x_data, y_data, r_data, diff_data

#------------spline--------------
def hermite_spline(x1,y1,g1,x2,y2,g2):
    A = np.array([
        [x1**3,x1**2,x1,1],
        [x2**3,x2**2,x2,1],
        [3*x1**2,2*x1,1,0],
        [3*x2**2,2*x2,1,0]
    ])

    B = np.array([y1,y2,g1,g2])
    return np.linalg.solve(A,B)

def substitute(coefficient,x):
    return coefficient[0]*x**3 + coefficient[1]*x**2 + coefficient[2]*x + coefficient[3]

def fit(x, y, k = 1):
    """
        k 는 기울기 강도 커질수록 언더피팅을 유발함
    """
    if len(x) < 5:
        return [], []
    x = list(x)
    y = list(y)
    xi = np.array([x[0]] + x + [x[-1]])
    yi = np.array([y[0]] + y + [y[-1]])
    poly_x = []
    poly_y = []
    gx = (xi[2:] - xi[:-2])/k
    gy = (yi[2:] - yi[:-2])/k
    for i in range(len(x)-1):
        result_x = hermite_spline(0,x[i],gx[i],1,x[i+1],gx[i+1])
        result_y = hermite_spline(0,y[i],gy[i],1,y[i+1],gy[i+1])

        poly_x.append(result_x)
        poly_y.append(result_y)

    return poly_x, poly_y

def get_lines(poly_x,sample_num = 10):
    sample = np.linspace(0,1,sample_num,endpoint= False)
    result_x = []

    for i in range(len(poly_x)):
        x = substitute(poly_x[i],sample)
        result_x += list(x)

    return np.array(result_x)

def differentiate(poly_array):
    diff_array = []
    for poly in poly_array:
        diff_array.append(np.array([0,3*poly[0],2*poly[1],poly[2]]))

    return diff_array

#------------sub pixel edge--------------
def calculate_pixel(img, coor):
    """
    Calculates the pixel value at a given coordinate of an image.

    Args:
        img: A numpy array representing the image.
        coor: A tuple or list representing the (x, y) coordinate of the pixel.

    Returns:
        A float value representing the pixel value at the given coordinate.
    """
    min_x = int(np.floor(coor[0]))
    min_y = int(np.floor(coor[1]))
    max_x = min_x + 1
    max_y = min_y + 1

    if min_x<0 or min_x>img.shape[1] -1:
        return 0
    if max_x<0 or max_x>img.shape[1] -1:
        return 0
    if min_y<0 or min_y>img.shape[0] -1:
        return 0
    if max_y<0 or max_y>img.shape[0] -1:
        return 0
    

    a = coor[1] - min_y
    
    b = 1 - a
    p = coor[0] - min_x
    q = 1 - p

    A = img[min_y][min_x]
    B = img[max_y][min_x]
    C = img[max_y][max_x]
    D = img[min_y][max_x]

    pixel_val = q*(b*A+a*B) + p*(b*D + a*C)

    return pixel_val

def get_edge(img, center_coordinate, center_tan, vessel_radius, radius_edge_width_ratio = 0.5, sampling_num = 10, power_factor = 2, profile = False):
    """
    Get the endpoints of a vessel segment by calculating the vessel's edge profile and its mass center.

    Args:
        img (ndarray): Image data.
        center_coordinate (tuple): The center coordinates of the vessel.
        center_tan (float): The tangent of the vessel at its center.
        vessel_radius (float): The radius of the vessel.
        sampling_num (int, optional): The number of samples used to calculate the edge profile. Defaults to 10.
        power_factor (int, optional): The exponent used to determine the intensity of the edge profile. Defaults to 2.
        profile (bool, optional): If True, returns the edge profile; otherwise, returns the endpoint coordinates. Defaults to False.

    Returns:
        tuple or ndarray: The coordinates of the endpoints of the vessel segment or the edge profile, depending on the value of the "profile" argument.
    """

    #center_coordinate = (center_coordinate[0] + 0.5, center_coordinate[1] + 0.5)
    
    #1. edge_profile 가져오기
    lenth_per_pixel = 1
    radius_in_pixel = vessel_radius * lenth_per_pixel
    
    
    # 1_a. 구해야할 픽셀의 좌표를 구합니다. (1 -> 윗쪽에 있는 프로파일, 2 -> 아랫쪽에 있는 프로파일)
    edge_width = radius_in_pixel * radius_edge_width_ratio
    edge_start_lenth = radius_in_pixel * (1 - radius_edge_width_ratio/2)

    sample = edge_start_lenth + np.linspace(0, edge_width, sampling_num)
    angle = (np.arctan(center_tan))  #+ np.pi/2 ) 

    x1 = center_coordinate[0] + (sample.copy() * np.cos(angle))
    y1 = center_coordinate[1] + (sample.copy() * np.sin(angle))
    x2 = center_coordinate[0] - (sample.copy() * np.cos(angle))
    y2 = center_coordinate[1] - (sample.copy() * np.sin(angle))

    # 1_b. 주변의 픽셀을 이용하여 픽셀 값을 찾습니다.
    edge_profile_1 = []
    edge_profile_2 = []

    for i in range(len(sample)): #c++로 변환 필요 없을 시 벡터라이징 해야함
        edge_profile_1.append(calculate_pixel(img, [x1[i],y1[i]]))
        edge_profile_2.append(calculate_pixel(img, [x2[i],y2[i]]))
 
    profile1 = np.array(edge_profile_1)
    profile2 = np.array(edge_profile_2)

    if profile:
        return profile1, profile2

    #2. gradient 및 weight계산
    g1 = profile1[1:] - profile1[:-1]
    g2 = profile2[1:] - profile2[:-1]
    w1 = g1 ** power_factor
    w2 = g2 ** power_factor

    
    l1 = 0
    l2 = 0
    for i in range(len(w1)):
        l1 += (i + 0.5 * edge_width / sampling_num) * w1[i]
        l2 += (i + 0.5 * edge_width / sampling_num) * w2[i]
    l1 /= w1.sum()
    l2 /= w2.sum()
    #l2 = edge_width - l2

    #4. root mean squares of the differences (RMSD) 에 따른 가감
    pass

    #5. 원래 좌표로 환산
    edge1 = edge_start_lenth + l1/len(w2) * edge_width
    edge2 = edge_start_lenth + l2/len(w2) * edge_width
    edge_coor1 = (center_coordinate[0] + edge1 * np.cos(angle), center_coordinate[1] + edge1 * np.sin(angle))
    edge_coor2 = (center_coordinate[0] - edge2 * np.cos(angle), center_coordinate[1] - edge2 * np.sin(angle))

    return edge_coor1, edge_coor2
        

