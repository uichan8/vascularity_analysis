#ifndef branch_vectorization_hpp
#define branch_vectorization_hpp

typedef std::vector<std::vector<bool>> Mask;

#include "opencv2/opencv.hpp"
#include <vector>

int count_boundary_point(Mask& target_line_mask, cv::Point2d point);

cv::Point2d find_end_point(Mask& target_line_mask);

std::tuple<std::vector<int>, std::vector<int>> find_track_path(Mask& target_line_mask, cv::Point2d point);

std::tuple<std::vector<int>, std::vector<int>> sort_points(const Mask& target_line_mask);
#endif 
std::vector<double> hermite_spline(double x1, double y1, double g1, double x2, double y2, double g2);

double substitute(std::vector<double> coefficients, double x);

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> fit(std::vector<double> x, std::vector<double> y, double k = 1);

std::vector<double> get_lines(std::vector<std::vector<double>> poly_x, int sample_num = 10);

std::vector<std::vector<double>> differentiate(std::vector<std::vector<double>> poly_array);