import cv2
import numpy as np

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

