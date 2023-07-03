#ifndef vessel_width_detection_hpp
#define vessel_width_detection_hpp

#include "opencv2/opencv.hpp"
#include <vector>

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> mask_witdth_detection(cv::Mat& seg_mask, std::vector<cv::Point2d> pts_arr);

double calculate_pixel(cv::Mat& img, cv::Point2d coor);

std::vector<cv::Point2d> get_edge(cv::Mat& img, cv::Point2d center_coordinate, double center_tan, double vessel_radius);

#endif