#include "opencv2/opencv.hpp"
#include "bifur_vectorization.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <set>
#include <algorithm>

#define M_PI 3.14159265358979323846

using namespace std;


Circle::Circle(int max_r) {
	MAX_R = max_r;
	for (int i = 0; i <= MAX_R; i++) {
		if (i % 2 == 1) {
			circle_edge_list.push_back(get_circle(i));
			circle_mask_list.push_back(get_circle_mask(i));
		}
	}
}
/*
bool Circle::comparePair(pair<int, int>& p1, pair<int, int>& p2) {
	return angle(0, 0, p1.first, p1.second) < angle(0, 0, p2.first, p2.second);
}
*/

double Circle::angle(double x_center, double y_center, double x, double y) {
	double dx = x - x_center;
	double dy = y - y_center;
	double radians = atan2(dy, dx);
	double degrees = radians * 180 / M_PI;

	return degrees;
}


vector<pair<int, int>> Circle::get_circle(int r) {
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
	sort(circle_coor.begin(), circle_coor.end(), [&](const pair<int, int>& p1, const pair<int, int>& p2) {
		return angle(0, 0, p1.first, p1.second) < angle(0, 0, p2.first, p2.second);
		});

	return circle_coor;
}

vector<vector<int>> Circle::get_circle_mask(int radius) {
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

vector<vector<pair<int, int>>>& Circle::get_circle_edge_list() {
	return circle_edge_list;
}
vector<vector<vector<int>>>& Circle::get_circle_mask_list() {
	return circle_mask_list;
}
int Circle::get_MAX_R() {
	return MAX_R;
}

vector<int> get_pixel_values(cv::Mat& mask, vector<pair<int, int>>& coordinates) {
	vector<int> pixel_values;

	for (const auto& coordinate : coordinates) {
		int x = coordinate.first;
		int y = coordinate.second;

		if (x >= 0 && x < mask.cols && y >= 0 && y < mask.rows) {
			pixel_values.push_back(mask.at<uchar>(y, x));
		}
		else {
			pixel_values.push_back(0);
		}
	}

	if (!pixel_values.empty()) {
		pixel_values.push_back(pixel_values[0]);
	}

	return pixel_values;
}

vector<vector<int>> find_branch_mask(cv::Mat& mask, int x, int y, Circle C) {

	int i = 0;
	vector<vector<pair<int, int>>> circle_edge_list = C.get_circle_edge_list();
	vector<vector<vector<int>>> circle_mask_list = C.get_circle_mask_list();

	for (const auto& circle_edge : circle_edge_list) {
		int r = 2 * i + 1;
		vector<pair<int, int>> coor;
		for (const auto& coordinate : circle_edge) {
			int coor_x = coordinate.first + x;
			int coor_y = coordinate.second + y;
			coor.push_back(make_pair(coor_x, coor_y));
		}

		vector<int> l = get_pixel_values(mask, coor);

		int sum_val = 0;

		for (int j = 1; j < l.size(); j++) {
			if (l[j] != l[j - 1]) {
				sum_val += 1;
			}
		}
		if (sum_val > 5 || r == C.get_MAX_R()) {
			break;
		}
		i += 1;
	}
	vector<vector<int>> branch_mask;
	int r = 2 * i + 1;

	for (int j = y - r; j <= y + r; j++) {
		vector<int> row;
		for (int k = x - r; k <= x + r; k++) {
			if (j >= 0 && j < mask.rows && k >= 0 && k < mask.cols) {
				row.push_back(mask.at<uchar>(j, k) * circle_mask_list[i][j - y + r][k - x + r]);
			}
		}
		branch_mask.push_back(row);
	}
	return branch_mask;
}

void draw_line(cv::Mat& mask, cv::Point point1, cv::Point point2, char color, int thickness) {
	int height = mask.rows;
	int width = mask.cols;
	point1.x = (point1.x < 0) ? 0 : (point1.x >= width) ? width - 1 : point1.x;
	point1.y = (point1.y < 0) ? 0 : (point1.y >= height) ? height - 1 : point1.y;
	point2.x = (point2.x < 0) ? 0 : (point2.x >= width) ? width - 1 : point2.x;
	point2.y = (point2.y < 0) ? 0 : (point2.y >= height) ? height - 1 : point2.y;

	vector<float> x_values(100);
	vector<float> y_values(100);
	for (int i = 0; i < 100; i++) {
		float t = static_cast<float>(i) / 99;
		x_values[i] = static_cast<float>(point1.x) + t * (static_cast<float>(point2.x) - static_cast<float>(point1.x));
		y_values[i] = static_cast<float>(point1.y) + t * (static_cast<float>(point2.y) - static_cast<float>(point1.y));
	}

	if (color == 'r') {
		for (int i = -thickness / 2; i <= thickness / 2; i++) {
			for (int j = 0; j < 100; j++) {
				int y = static_cast<int>(round(y_values[j])) + i;
				int x = static_cast<int>(round(x_values[j]));
				if (y >= 0 && y < height && x >= 0 && x < width) {
					mask.at<cv::Vec3b>(y, x)[0] = 0;
				}
			}
		}
	}
	else if (color == 'b') {
		for (int i = -thickness / 2; i <= thickness / 2; i++) {
			for (int j = 0; j < 100; j++) {
				int y = static_cast<int>(round(y_values[j]));
				int x = static_cast<int>(round(x_values[j])) + i;
				if (y >= 0 && y < height && x >= 0 && x < width) {
					mask.at<cv::Vec3b>(y, x)[2] = 0;
				}
			}
		}
	}
}