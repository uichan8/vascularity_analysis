#include "vessel_width_detection.hpp"
#include "opencv2/opencv.hpp"

#include <stdexcept>
#include <cmath>
#include <vector>
#define M_PI 3.14159265358979323846

using namespace std;

tuple<vector<double>, vector<double>, vector<double>, vector<double>>mask_witdth_detection(cv::Mat& seg_mask, vector<cv::Point2d> pts_arr) {
    vector<double> x1_edge, x2_edge, y1_edge, y2_edge;
    double x, y, diff_x, diff_y, normal_x, normal_y;

    for (int i = 0; i < pts_arr.size(); i += 2) {
        if (i == pts_arr.size() - 1) {
            x = pts_arr[i].x;
            y = pts_arr[i].y;
            diff_x = pts_arr[i].x - pts_arr[i - 1].x;
            diff_y = pts_arr[i].y - pts_arr[i - 1].y;
        }
        else {
            x = pts_arr[i].x;
            y = pts_arr[i].y;
            diff_x = pts_arr[i + 1].x - pts_arr[i].x;
            diff_y = pts_arr[i + 1].y - pts_arr[i].y;
            if (diff_x == 0 && diff_y == 0) {
                diff_x = pts_arr[i + 2].x - pts_arr[i].x;
                diff_y = pts_arr[i + 2].y - pts_arr[i].y;
            }
            normal_x = diff_y;
            normal_y = -diff_x;
        }


        // if (normal_x == 0 && normal_y == 0) {
        //     continue;
        // }

        double x1 = x;
        double x2 = x;
        double y1 = y;
        double y2 = y;

        while (x1 < seg_mask.rows && y1 < seg_mask.cols) {
            double x_prime = x1 + normal_x;
            double x_pprime = x2 - normal_x;
            double y_prime = y1 + normal_y;
            double y_pprime = y2 - normal_y;


            if (static_cast<int>(seg_mask.at<uchar>(static_cast<int>(x_prime), static_cast<int>(y_prime))) == 0 && static_cast<int>(seg_mask.at<uchar>(static_cast<int>(x_pprime), static_cast<int>(y_pprime))) == 0) {
                x1_edge.push_back(x_prime - 0.5);
                x2_edge.push_back(x_pprime + 0.5);
                y1_edge.push_back(y_prime - 0.5);
                y2_edge.push_back(y_pprime + 0.5);
                break;
            }

            if (static_cast<int>(seg_mask.at<uchar>(static_cast<int>(x_prime), static_cast<int>(y_prime))) == 0) {
                x_prime = x1;
                y_prime = y1;
            }
            if (static_cast<int>(seg_mask.at<uchar>(static_cast<int>(x_pprime), static_cast<int>(y_pprime))) == 0) {
                x_pprime = x2;
                y_pprime = y2;
            }
            x1 = x_prime;
            x2 = x_pprime;
            y1 = y_prime;
            y2 = y_pprime;
        }
    }

    return make_tuple(x1_edge, x2_edge, y1_edge, y2_edge);
}

/**
 @brief Calculates the pixel value at the given coordinate in the input image.

 @param img The input image.
 @param coor The coordinate of the pixel.

 @return The pixel value at the given coordinate.
 */
double calculate_pixel(cv::Mat& img, cv::Point2d coor) {
    /**
    Calculates the pixel value at a given coordinate of an image.

    Args:
        img: A numpy array representing the image.
        coor: A tuple or list representing the (x, y) coordinate of the pixel.

    Returns:
        A float value representing the pixel value at the given coordinate.
    */
    int min_x = coor.x;
    int min_y = coor.y;
    int max_x = min_x + 1;
    int max_y = min_y + 1;

    if (min_x < 0 || min_x >= img.cols - 1 || min_y < 0 || min_y >= img.rows - 1)
        throw out_of_range("The given point is out of image range.");


    double a = coor.y - min_y;
    double b = 1 - a;
    double p = coor.x - min_x;
    double q = 1 - p;

    uchar A = img.at<uchar>(min_x, min_y);
    uchar B = img.at<uchar>(min_x, max_y);
    uchar C = img.at<uchar>(max_x, min_y);
    uchar D = img.at<uchar>(max_x, max_y);

    double pixel_val = q * (b * A + a * B) + p * (b * D + a * C);

    return pixel_val;
}


/**
 * @brief Get the edge coordinates of a branch in an image.
 *
 * @param img The input image.
 * @param center_coordinate The center point of the branch.
 * @param center_tan The tangent of the angle between the branch and the horizontal axis.
 * @param branch_radius The radius of the branch.
 *
 * @return A vector of two points representing the coordinates of the edges of the branch.
 */
vector<cv::Point2d> get_edge(cv::Mat& img, cv::Point2d center_coordinate, double center_tan, double branch_radius) {
    /**
    Get the endpoints of a branch segment by calculating the branch's edge profile and its mass center.

    Args:
        img (ndarray): Image data.
        center_coordinate (tuple): The center coordinates of the branch.
        center_tan (float): The tangent of the branch at its center.
        branch_radius (float): The radius of the branch.
        sampling_num (int, optional): The number of samples used to calculate the edge profile. Defaults to 10.
        power_factor (int, optional): The exponent used to determine the intensity of the edge profile. Defaults to 2.
        profile (bool, optional): If True, returns the edge profile; otherwise, returns the endpoint coordinates. Defaults to False.

    Returns:
        tuple or ndarray: The coordinates of the endpoints of the branch segment or the edge profile, depending on the value of the "profile" argument.
    */
    const double edge_width = 3;
    const int sampling_num = 40;
    const int P = 2; //power_factor

    //1. edge_profile 가져오기
    double edge_start_point = branch_radius - edge_width / 2;
    double sample[sampling_num];
    for (int i = 0; i < sampling_num; i++)
        sample[i] = edge_start_point + edge_width / (sampling_num - 1) * i;

    double x1[sampling_num], y1[sampling_num], x2[sampling_num], y2[sampling_num];
    double angle = atan(center_tan) + M_PI / 2;
    for (int i = 0; i < sampling_num; i++) {
        x1[i] = center_coordinate.x + (sample[i] * cos(angle));
        y1[i] = center_coordinate.y + (sample[i] * sin(angle));
        x2[i] = center_coordinate.x - (sample[i] * cos(angle));
        y2[i] = center_coordinate.y - (sample[i] * sin(angle));
    }


    double edge_profile_1[sampling_num], edge_profile_2[sampling_num];
    for (int i = 0; i < sampling_num; i++) {
        edge_profile_1[i] = calculate_pixel(img, cv::Point2d(x1[i], y1[i]));
        edge_profile_2[i] = calculate_pixel(img, cv::Point2d(x2[i], y2[i]));
    }

    //2. gradient 및 weight계산
    double w1, w2, w1_s = 0, w2_s = 0;
    double l1 = 0, l2 = 0;
    double half_block_w = edge_width / sampling_num * 0.5;
    //3. 질량 중심 구하기
    for (int i = 0; i < sampling_num - 1; i++) {
        w1 = pow((edge_profile_1[i + 1] - edge_profile_1[i]), P);
        w2 = pow((edge_profile_2[i + 1] - edge_profile_2[i]), P);
        w1_s += w1;    w2_s += w2;
        l1 += w1 + half_block_w;
        l2 += w2 + half_block_w;
    }
    l1 /= w1_s;   l2 /= w2_s;

    //원래 좌표로 환산
    double edge1 = edge_start_point + l1 / (sampling_num - 1) * edge_width;
    double edge2 = edge_start_point + l2 / (sampling_num - 1) * edge_width;
    cv::Point2d coor1(center_coordinate.x + edge1 * cos(angle), center_coordinate.y + edge1 * sin(angle));
    cv::Point2d coor2(center_coordinate.x - edge2 * cos(angle), center_coordinate.y - edge2 * sin(angle));

    vector<cv::Point2d> edge_coor = { coor1, coor2 };
    return edge_coor;
}