#include "opencv2/opencv.hpp"
#include "bifur_vectorization.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <set>

#define M_PI 3.14159265358979323846

using namespace std;

class Circle {

public:
	vector<vector<pair<int, int>>> circle_edge_list;
	vector<vector<vector<int>>> circle_mask_list;

	Circle(int max_r) {
		for (int i = 0; i <= max_r; i++) {
			if (i % 2 == 1) {
				circle_edge_list.push_back(get_circle(i));
				circle_mask_list.push_back(get_circle_mask(i));
			}
		}
	}

	static double angle(double x_center, double y_center, double x, double y) {
		double dx = x - x_center;
		double dy = y - y_center;
		double radians = atan2(dy, dx);
		double degrees = radians * 180 / M_PI;
		
		return degrees;
	}

	bool comparePair(const pair<int, int>& p1, const pair<int, int>& p2) {
		return angle(0, 0, p1.first, p1.second) < angle(0, 0, p2.first, p2.second);
	}
	
	vector<pair<int, int>> get_circle(int r) {
		int x = 0;
		int y = r;
		int d = 3 - 2 * r;
		set<pair<int, int>> pixels;

		while (x <= y) {
			pixels.insert(make_pair(x, y));
			pixels.insert(make_pair(y, x));
			pixels.insert(make_pair(x, y));
			pixels.insert(make_pair(y, x));
			pixels.insert(make_pair(x, y));
			pixels.insert(make_pair(y, x));
			pixels.insert(make_pair(x, y));
			pixels.insert(make_pair(y, x));

			if (d < 0) {
				d = d + 4 * x + 6;
			}
			else {
				d = d + 4 * (x - y) + 10;
				y -= 1;
			}
			x += 1;
		}

		for (const auto& pixel : pixels) {
			int x = pixel.first;
			int y = pixel.second;

			pixels.insert(make_pair(-x, y));
			pixels.insert(make_pair(x, -y));
			pixels.insert(make_pair(-x, -y));
		}

		vector<pair<int, int>> circle_coor(pixels.begin(), pixels.end());
		sort(circle_coor.begin(), circle_coor.end(), comparePair);

		return circle_coor;
	}

	vector<vector<int>> get_circle_mask(int radius) {
		int mask_shape = 2 * radius + 1;
		vector<vector<int>> mask(mask_shape, vector<int>(mask_shape, 0));

		for (int y = 0; y < mask_shape; y++) {
			for (int x = 0; x < mask_shape; x++) {
				if (pow(x - radius, 2) + pow(y - radius, 2) <= pow(radius, 2)) {
					mask[y][x] = 1;
				}
			}
		}
		return mask;
	}

	const vector<vector<pair<int, int>>>& get_circle_edge_list() {
		return circle_edge_list;
	}
	const vector<vector<vector<int>>>& get_circle_mask_list() {
		return circle_mask_list;
	}
	
};

vector<int> get_pixel_values(const cv::Mat& mask, const vector<pair<int, int>>& coordinates) {
	vector<int> pixel_values;

	for (const auto& coordinate : coordinates) {
		int x = coordinate.first;
		int y = coordinate.second;

		if (x >= 0 && x < mask.cols && y >= 0 && y < mask.rows) {
			pixel_values.push_back(mask.at<int>(y, x));
		}
		else {
			pixel_values.push_back(0);
		}
	}

	pixel_values.push_back(pixel_values[0]);
	return pixel_values;
}

vector<vector<int>> find_branch_mask(cv::Mat& mask, int x, int y,const Circle& C) {
	cv::Mat dilated_mask;
	cv::dilate(mask, dilated_mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

	int i = 0;
	int MAX_R = 19;

	for (const auto& circle_edge : C.circle_edge_list) {
		int r = 2 * i + 1;
		vector<pair<int, int>> coor;
		for (const auto& coordinate : circle_edge) {
			int coor_x = coordinate.first + x;
			int coor_y = coordinate.second + y;
			coor.push_back(make_pair(coor_x, coor_y));
		}

		vector<int> l = get_pixel_values(mask, coor);

		int sum_val = 0;

		for (int j = 0; j < l.size(); j++) {
			if (l[j] != l[j - 1]) {
				sum_val += 1;
			}
		}
		if (sum_val > 5 || r == MAX_R) {
			break;
		}
		i += 1;
	}
	vector<vector<int>> branch_mask;
	int r = 2 * i + 1;

	for (int j = y - r; j <= y + r; j++) {
		vector<int> row;
		for (int k = x - r; k <= x + r; k++) {
			row.push_back(mask.at<int>(j, k) * C.circle_mask_list[i][j - y + r][k - x + r]);
		}
		branch_mask.push_back(row);
	}
	return branch_mask;
}