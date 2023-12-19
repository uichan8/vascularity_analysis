
import cv2
import numpy as np
import find_branch
import subpixel_edge
import bspline_regression
from PIL import Image
from time import time
import scipy.linalg
from tqdm import tqdm

from uniform_bspline import UniformBSpline

import matplotlib.pyplot as plt
   
# main
def main(mask_path, img_path, param_w = [1], solver_type = 'dn', max_num_iterations = 100, ratio = 0.5, min_radius = 1e-9, max_radius = 1e12, num_samples = 1024, initial_radius = 1e4, degree = 2, dim= 2, num_init_points = 32):
    example_mask = np.array(Image.open(mask_path))
    mask = example_mask
    real_img = np.array(Image.open(img_path))

    example_mask = (cv2.cvtColor(example_mask, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    partion_mask, cl_mask = find_branch.segment_partion(example_mask)
    retvals, labels, stats, centroids = cv2.connectedComponentsWithStats(cl_mask)
    st_time = time()


    result_x = []
    result_y = []
    result_x2 = []
    result_y2 = []
    result_center_x = []
    result_center_y = []

    blur_img = cv2.GaussianBlur(real_img[:,:,1], (5,5), 3)

    
    # for vessel_num in range(labels.shape[2]):

    for label_num in tqdm(range(1,np.max(labels)+1)):
        all_X_data, all_Y_data  = np.where(labels== label_num)
        flag = 0
        
        for i in range(len(all_Y_data)-1):
            if (all_Y_data[i+1] - all_Y_data[i]) > 1:
                flag = 1

        all_X_data = [all_X_data.tolist()]
        all_Y_data = [all_Y_data.tolist()]
        
        for j in range(len(all_X_data)):
            #st_time = time()
            n_data_points = len(all_X_data[j])
            #cur_mask = np.zeros((nX, nY), dtype=np.uint8)
            
            # sjgo add this code - adjustment control point
            if int(n_data_points * ratio) < 5:
                n_cont_points = 5
            else:
                n_cont_points = int(n_data_points * ratio)

            if dim == 2:
                if flag == 0:
                    Y = np.c_[all_X_data[j], all_Y_data[j]]
                else:
                    Y = np.c_[all_Y_data[j],all_X_data[j]]
                    Y = Y.tolist()
                    Y.sort(key = lambda x : x[0])
                    Y = np.array(Y)
            else:
                Y = np.c_[all_X_data[j], all_Y_data[j], np.linspace(0.0, 1.0, n_data_points)]

            t = np.int_(np.linspace(0.0, n_data_points-1, n_cont_points)[:, np.newaxis])
            X_t = []

            for i_t in range(len(t)):
                X_t.append([Y[t[i_t]][0][0], Y[t[i_t]][0][1]])

            X = np.asarray(X_t)
            c = UniformBSpline(degree, n_cont_points, dim)


            # Set `w`.
            if np.any(np.asarray(param_w) < 0):
                raise ValueError('w <= 0.0 (= {})'.format(param_w))
            if len(param_w) == 1:
                w = np.empty((n_data_points, dim), dtype=float)
                w.fill(param_w[0])
            elif len(param_w) == dim:
                w = np.tile(param_w, (n_data_points, 1))
            else:
                raise ValueError('len(w) is invalid (= {})'.format(len(param_w)))

            # Initialise `u`.
            u0 = c.uniform_parameterisation(num_init_points)
            D = scipy.spatial.distance.cdist(Y, c.M(u0, X))
            u = u0[D.argmin(axis=1)]
            

            # print('UniformBSplineLeastSquaresOptimiser Output:')
            (u1, X1,
            has_converged,
            states, num_iterations, time_taken
            ) = bspline_regression.UniformBSplineLeastSquaresOptimiser(c, solver_type).minimise(
                Y, w, u, X,
                max_num_iterations = max_num_iterations,
                min_radius=min_radius,
                max_radius=max_radius,         
                initial_radius=initial_radius,
                return_all=True)

            c = UniformBSpline(degree, n_cont_points, dim)


            num_samples = n_data_points * 10

            # Evaluate points on the contour.
            # main func.
            fit_pts = (c.M(c.uniform_parameterisation(num_samples), X1)).astype(float)

            for i in range(len(Y)):
                if Y[i,0] < np.min(fit_pts[:,0]):
                    fit_pts = np.insert(fit_pts, 0, Y[i], axis=0)
                if Y[i,0] > np.max(fit_pts[:,0]):
                    fit_pts = np.append(fit_pts, [Y[i]], axis=0)


            # sjgo add this code - pixel-wise linear interpolation
            c_i = 0
            l = []
            c = []
            for j in range(len(fit_pts) -1):
                l_i = np.sqrt(pow(fit_pts[j+1][0]- fit_pts[j][0], 2) + pow(fit_pts[j+1][1] - fit_pts[j][1] ,2))
                l.append(l_i)
                c_i += l_i
                c.append(c_i)

            x_idx = np.linspace(0, int(c[-1]), int(c[-1]) + int(c[-1])+1)
            #x_idx = np.arange(0, int(c[-1]), 4)
            new_x = []
            new_y = []
            found = []

            #find index
            for j in range(len(x_idx)):
                min = 1000
                found_i = 0
                for i in range(len(c)):
                    tmp = c[i] - x_idx[j]
                    if abs(min) > abs(tmp):
                        min = tmp
                        near = c[i]
                        found_i = i
                    if c[i] < x_idx[j]:
                        found_i = i+1
                found.append(found_i)

            #re-sampling

            for i, f_idx in enumerate(found):
                if i == 0 :
                    new_x.append(fit_pts[0][0])
                    new_y.append(fit_pts[0][1])
                else:
                    if f_idx == x_idx[i]:
                        alpha = 0
                    else:
                        alpha = (x_idx[i] - c[f_idx-1])/(c[f_idx] - c[f_idx-1])

                        #alpha = (x_idx[i] - c[f_idx])/(c[f_idx+1] - c[f_idx])

                    new_x.append(float(fit_pts[f_idx+1][0] * alpha) + (1 - alpha) * fit_pts[f_idx][0])
                    new_y.append(fit_pts[f_idx+1][1] * alpha + (1 - alpha) * fit_pts[f_idx][1])
                    
            new_pts = np.c_[new_x, new_y]
            # r_new_pts = np.round(new_pts, 0).astype(np.int64)
            r_new_pts = np.round(new_pts, 2)

            if flag == 1:
                tmp = r_new_pts[:,0].copy()
                r_new_pts[:,0] = r_new_pts[:,1]
                r_new_pts[:,1] = tmp

            edge_x = np.array([])
            edge_x2 = np.array([])
            edge_y = np.array([])
            edge_y2 = np.array([])
            
            edge_x, edge_x2, edge_y, edge_y2 = subpixel_edge.width_detection(example_mask, r_new_pts,edge_x, edge_x2, edge_y, edge_y2)

            x_cen = (edge_x+edge_x2)/2
            y_cen = (edge_y+edge_y2)/2
            center_tan = (edge_y - edge_y2)/(edge_x-edge_x2+1e-12)
            vessel_w = np.sqrt((edge_y - edge_y2)**2+(edge_x-edge_x2)**2)/2

            if vessel_w.shape[0] == 0:
                continue

            vessel_s = np.zeros_like(vessel_w)

            vessel_s[:-1] += vessel_w[1:]
            vessel_s[1:] += vessel_w[:-1]
            vessel_s += vessel_w
            vessel_s[1:-1]/=3
            vessel_s[0] /= 2
            vessel_s[-1] /= 2

            for i in range(len(vessel_s) - 1):
                if vessel_s[i]*0.9 > vessel_s[i+1]:
                    vessel_s[i+1] = vessel_s[i]*0.9
                elif vessel_s[i]*1.1 < vessel_s[i+1]:
                    vessel_s[i+1] = vessel_s[i]*1.1


            for i in range(len(x_cen)):
                a,b = subpixel_edge.get_edge(blur_img,(x_cen[i],y_cen[i]),center_tan[i],vessel_s[i],radius_edge_width_ratio = 4/vessel_s[i], sampling_num = 50, power_factor = 2)
                a2,b2 = subpixel_edge.get_edge(blur_img,(x_cen[i],y_cen[i]),center_tan[i],vessel_w[i],radius_edge_width_ratio = 4/vessel_w[i], sampling_num = 50, power_factor = 2)
                result_x.append(a[0])
                result_x.append(b[0])
                result_y.append(a[1])
                result_y.append(b[1])

                result_x2.append(a2[0])
                result_x2.append(b2[0])
                result_y2.append(a2[1])
                result_y2.append(b2[1])
                
                result_center_x.append((a[0] + b[0])/2)
                result_center_y.append((a[1] + b[1])/2)

        

    



    ed_time = time()
    print(ed_time - st_time)
    #Image.fromarray(cur_mask).save('centerline_spline_fixed.png')
    plt.imshow(real_img)
    plt.scatter(result_y,result_x,c='r',s = 5, label = 'smoothing')
    plt.scatter(result_y2,result_x2, c='b', s = 5, label = 'origin')
    #plt.scatter(result_center_y, result_center_x)
    plt.legend()
    plt.savefig('result_real_img.png')
    plt.show()

    plt.imshow(mask)
    plt.scatter(result_y,result_x,c='g',s = 5, label = 'smoothing')
    plt.scatter(result_y2,result_x2, c='orange', s = 5, label = 'origin')
    #plt.scatter(result_center_y, result_center_x)
    plt.legend()
    plt.savefig('result_mask_img.png')
    plt.show()

    print(ed_time - st_time)

    np.save('result_x.npy',result_x)
    np.save('result_y.npy',result_y)
    np.save('result_x2.npy',result_x2)
    np.save('result_y2.npy',result_y2)


    # plt.imshow(img)
    # plt.scatter(cur_mask)
            
            #f.close()


if __name__ == '__main__':
    mask_path = "img/mask.png"
    img_path = "img/real.png"
    main(mask_path, img_path)
