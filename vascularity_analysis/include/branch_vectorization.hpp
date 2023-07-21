#ifndef branch_vectorization_hpp
#define branch_vectorization_hpp

#include "opencv2/opencv.hpp"
#include <vector>

int count_boundary_point(cv::Mat target_line_mask, cv::Point2d point);

cv::Point2d find_end_point(cv::Mat target_line_mask);

std::vector<cv::Point2d> find_track_path(cv::Mat target_line_mask, cv::Point2d point);

//std::tuple<std::vector<int>, std::vector<int>> sort_points(const Mask& target_line_mask);
std::vector<cv::Point2d> sort_points(const cv::Mat& target_line_mask);

std::vector<double> hermite_spline(double x1, double y1, double g1, double x2, double y2, double g2);

double substitute(std::vector<double> coefficients, double x);

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> fit(std::vector<double> x, std::vector<double> y, double k = 1);

std::vector<double> get_lines(std::vector<std::vector<double>> poly_x, int sample_num = 10);

std::vector<std::vector<double>> differentiate(std::vector<std::vector<double>> poly_array);

std::vector<double> simple_sampling(std::vector<double>& arr, int sparsity);

void delete_outliers(std::vector<double>& x_data, std::vector<double>& y_data, std::vector<double>& r_data, std::vector<double>& diff_data, double threshold = 3);

#endif 