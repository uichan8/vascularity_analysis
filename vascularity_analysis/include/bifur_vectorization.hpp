#ifndef bifur_vectorization_hpp
#define bifur_vectorization_hpp
#endif 

#include "opencv2/opencv.hpp"
#include <vector>

class Circle {
public:
	Circle(int max_r);

	std::vector<std::vector<int>> circle_edge_list;
	std::vector<std::vector<std::vector<int>>> circle_mask_list;

	static double angle(double x_center, double y_center, double x, double y);
	bool comparePair(const std::pair<int, int>& p1, const std::pair<int, int>& p2);

	std::vector<std::pair<int, int>> get_circle(int r);
	std::vector<std::vector<int>> get_circle_mask(int radius);


	const std::vector<std::vector<std::pair<int, int>>>& get_circle_edge_list();
	const std::vector<std::vector<std::vector<int>>>& get_circle_mask_list();


};

std::vector<int> get_pixel_values(const cv::Mat& mask, const std::vector<std::pair<int, int>>& coordinates);
std::vector<std::vector<int>> find_branch_mask(cv::Mat& mask, int x, int y, const Circle& C);
