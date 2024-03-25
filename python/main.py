import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import skeletonize


from tools import bifur
from tools import branch
from tools import struct

img_path = "../data/img/020_img.png"
mask_path = "../data/mask/020_mask.png"

#이미지 로드
mask = Image.open(mask_path)
mask = np.array(mask)
mask_r = mask[:,:,0]
mask_b = mask[:,:,2]
img = Image.open(img_path)
img = np.array(img)
blur_img = cv2.GaussianBlur(img[:,:,1], (5,5), 3)

#스켈레톤
skeleton_img = skeletonize(mask, method='lee')
skeleton_r = skeleton_img[:,:,0]
skeleton_b = skeleton_img[:,:,2]


#분기점 
junction_coor_r = np.where(bifur.find_bifur_points(skeleton_r > 0))
junction_coor_b = np.where(bifur.find_bifur_points(skeleton_b > 0))

#분기점 제거및 마스크 추출
center_r = []
for i in tqdm(range(len(junction_coor_r[0]))):
    jx = junction_coor_r[0][i]
    jy = junction_coor_r[1][i]
    center_r.append([jx, jy])
    branch_mask = bifur.find_bifur_mask(mask_r,jy,jx)
    S = branch_mask.shape[0]//2
    skeleton_r[jx-S:jx+S+1, jy-S:jy+S+1] = 0 
    


center_b = []
for i in tqdm(range(len(junction_coor_b[0]))):
    jx = junction_coor_b[0][i]
    jy = junction_coor_b[1][i]
    center_b.append([jx, jy])
    skeleton_b[jx-1:jx+2, jy-1:jy+2] = 0
    branch_mask = bifur.find_bifur_mask(mask_b,jy,jx)
    S = branch_mask.shape[0]//2
    skeleton_b[jx-S:jx+S+1, jy-S:jy+S+1] = 0



    #vertex 분리
retvals_r, labels_r, stats_r, _ = cv2.connectedComponentsWithStats(skeleton_r)
retvals_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(skeleton_b)

def draw_line(mask, point1, point2, color='r', thickness=2):
    # 이미지의 크기를 확인하여 선의 좌표를 유효 범위로 조정합니다.
    height, width = mask.shape[:2]
    point1 = np.clip(point1, [0, 0], [width-1, height-1])
    point2 = np.clip(point2, [0, 0], [width-1, height-1])

    # 선의 좌표를 이용하여 직선 방정식을 구합니다.
    x_values = np.linspace(point1[0], point2[0], num=100)
    y_values = np.linspace(point1[1], point2[1], num=100)

    # 굵기에 따라 선을 그립니다.
    if color == 'r':
        for i in range(-thickness // 2, thickness // 2 + 1):
            mask[np.round(y_values + i).astype(int), np.round(x_values).astype(int),0] = 0
    elif color == 'b':
        for i in range(-thickness // 2, thickness // 2 + 1):
            mask[np.round(y_values).astype(int), np.round(x_values + i).astype(int),2] = 0

vessel_informs = [(retvals_r,labels_r,mask[:,:,0],'r'),(retvals_b,labels_b,mask[:,:,2],'b')]


center_x = []
center_y = []

skel_x = []
skel_y = []

sub_edge_x = []
sub_edge_y = []
sub_edge_x2 = []
sub_edge_y2 = []

sampling_num = 1

branch_mask = mask.copy()
mixed_img = 0.5 * img.astype(np.float32)/255.0 + 0.5 * mask.astype(np.float32)/255.0

plt.imshow(img)

for vessel in vessel_informs:
    sub_edge_x = []
    sub_edge_y = []
    for i in range(1,vessel[0]):
        target_line = (vessel[1] == i).copy()
        sorted_x,sorted_y = branch.sort_points(target_line)

        skel_x += sorted_x
        skel_y += sorted_y

        #spline
        # spline = fit(sorted_x, sorted_y, 1.5)
        # x = get_lines(spline[0],sampling_num)
        # y = get_lines(spline[1],sampling_num)

        edge_x, edge_x2, edge_y, edge_y2 = branch.mask_width_detection(vessel[2],np.c_[sorted_x,sorted_y])

        if len(edge_x) > 4:
            draw_line(branch_mask, (edge_x[1],edge_y[1]), (edge_x2[1],edge_y2[1]), color = vessel[3])
            draw_line(branch_mask, (edge_x[-2],edge_y[-2]), (edge_x2[-2],edge_y2[-2]), color = vessel[3])

        # sampling edge point
        edge_x = branch.simple_sampling(edge_x, 2)
        edge_y = branch.simple_sampling(edge_y, 2)
        edge_x2 = branch.simple_sampling(edge_x2, 2)
        edge_y2 = branch.simple_sampling(edge_y2, 2)


        x_cen = (edge_x+edge_x2)/2
        y_cen = (edge_y+edge_y2)/2
        center_tan = (edge_y - edge_y2)/(edge_x-edge_x2+1e-12) #주의
        vessel_w = np.sqrt((edge_y - edge_y2)**2+(edge_x-edge_x2)**2)/2

        subcen_x = np.array([])
        subcen_y = np.array([])
        r = np.array([])

        #subpixel_localization
        for i in range(len(x_cen)):
            if mask[int(y_cen[i]),int(x_cen[i]),0] == 255 and mask[int(y_cen[i]),int(x_cen[i]),2] == 255:
                a,b = branch.get_edge(blur_img,(x_cen[i],y_cen[i]), center_tan[i],vessel_w[i],radius_edge_width_ratio = 1/vessel_w[i], sampling_num = 50, power_factor = 2)
                subcen_x = np.append(subcen_x, (a[0]+b[0])/2)
                subcen_y = np.append(subcen_y, (a[1]+b[1])/2)
                r = np.append(r, np.sqrt((a[1] - b[1])**2+(a[0]-b[0])**2)/2)
            else:
                a,b = branch.get_edge(blur_img,(x_cen[i],y_cen[i]), center_tan[i],vessel_w[i],radius_edge_width_ratio = 3/vessel_w[i], sampling_num = 50, power_factor = 2)
                subcen_x = np.append(subcen_x, (a[0]+b[0])/2)
                subcen_y = np.append(subcen_y, (a[1]+b[1])/2)
                r = np.append(r, np.sqrt((a[1] - b[1])**2+(a[0]-b[0])**2)/2)

        sampling_num = 5


        sub_spline = branch.fit(subcen_x,subcen_y,10000)
        spline_x = branch.get_lines(sub_spline[0],sampling_num)#.clip(0,mask.shape[1]-1)
        spline_y = branch.get_lines(sub_spline[1],sampling_num)#.clip(0,mask.shape[0]-1)
        spline_diff_x = branch.differentiate(sub_spline[0])
        spline_diff_y = branch.differentiate(sub_spline[1])
        spline_diff_poly = []
        for i in range(len(spline_diff_x)):
            spline_diff_poly.append(spline_diff_y[i]/(spline_diff_x[i] + 1e-9)) 
        spline_diff = branch.get_lines(spline_diff_poly,sampling_num)

        
        a = np.arange(0,len(r),1)
        r_spline = branch.fit(a,r,1.5)

        spline_r = branch.get_lines(r_spline[1],sampling_num)

        
        if len(spline_x) < 5:
            continue

        spline_x, spline_y, spline_r, spline_diff = branch.delete_outliers(spline_x, spline_y, spline_r, spline_diff , 2)

        angle = np.arctan(spline_diff) + np.pi/2

        cal_x = np.array(spline_x) + spline_r * np.cos(np.array(angle))
        cal_y = np.array(spline_y) + spline_r * np.sin(np.array(angle))
        cal_x1 = np.array(spline_x) + spline_r * np.cos(np.array(angle)+np.pi)
        cal_y1 = np.array(spline_y) + spline_r * np.sin(np.array(angle)+np.pi)

        center_x += spline_x.tolist()
        center_y += spline_y.tolist()

        sub_edge_x += cal_x.tolist()
        sub_edge_y += cal_y.tolist()
        sub_edge_x += cal_x1.tolist()
        sub_edge_y += cal_y1.tolist()

        # if len(edge_x) > 2:
        #     draw_line(branch_mask, (cal_y[0],cal_x[0]), (cal_y1[0],cal_x1[0]), color = vessel[3])
        #     draw_line(branch_mask, (cal_y[-1],cal_x[-1]), (cal_y1[-1],cal_x1[-1]), color = vessel[3])
    if vessel[3] == 'r':
        plt.scatter(sub_edge_x,sub_edge_y, c = 'orange', s = 0.1)
    else:
        plt.scatter(sub_edge_x,sub_edge_y, c = 'g', s = 0.1)
        

plt.scatter(center_x, center_y, c = 'white', s = 0.1)


retvals_r, labels_r, stats_r, _ = cv2.connectedComponentsWithStats(branch_mask[:,:,0])
retvals_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(branch_mask[:,:,2])

mask_edge = np.zeros_like(mask)
for pts in center_r:
    label = labels_r[int(pts[0]),int(pts[1])]
    if id != 0 and (labels_r==label).sum() < 2000:
        branch_edge = (labels_r==label).astype(np.uint8)*255
        kernel = np.ones((3, 3), np.uint8)
        kernel[0,0] = 0
        kernel[0,2] = 0
        kernel[2,0] = 0
        kernel[2,2] = 0
        dilated_mask = cv2.dilate(branch_edge, kernel, iterations=1)
        dilated_mask *= (mask[:,:,0]==0)
        mask_edge[:,:,0] += dilated_mask

for pts in center_b:
    label = labels_b[int(pts[0]),int(pts[1])]
    if id != 0 and (labels_b==label).sum() < 2000:
        branch_edge = (labels_b==label).astype(np.uint8)*255
        kernel = np.ones((3, 3), np.uint8)
        kernel[0,0] = 0
        kernel[0,2] = 0
        kernel[2,0] = 0
        kernel[2,2] = 0
        dilated_mask = cv2.dilate(branch_edge, kernel, iterations=1)
        dilated_mask *= (mask[:,:,2]==0)
        mask_edge[:,:,2] += dilated_mask

edge_r = np.where(mask_edge[:,:,0] > 0)
edge_b = np.where(mask_edge[:,:,2] > 0)

plt.scatter(edge_r[1],edge_r[0], c = 'r', s = 0.01)
plt.scatter(edge_b[1],edge_b[0], c = 'b', s = 0.01)

plt.show()