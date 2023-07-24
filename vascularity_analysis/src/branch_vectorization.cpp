#include "opencv2/opencv.hpp"
#include "branch_vectorization.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <stdexcept>
#include <numeric>

using namespace std;

int count_boundary_point(cv::Mat target_line_mask, cv::Point2d point) {
	int num_count = 0;

	num_count += (target_line_mask.at<uchar>(point.y - 1, point.x - 1) != 0) ? 1 : 0;
	num_count += (target_line_mask.at<uchar>(point.y - 1, point.x) != 0) ? 1 : 0;
	num_count += (target_line_mask.at<uchar>(point.y - 1, point.x + 1) != 0) ? 1 : 0;
	num_count += (target_line_mask.at<uchar>(point.y, point.x - 1) != 0) ? 1 : 0;
	num_count += (target_line_mask.at<uchar>(point.y, point.x + 1) != 0) ? 1 : 0;
	num_count += (target_line_mask.at<uchar>(point.y, point.x) != 0) ? 1 : 0;
	num_count += (target_line_mask.at<uchar>(point.y + 1, point.x - 1) != 0) ? 1 : 0;
	num_count += (target_line_mask.at<uchar>(point.y + 1, point.x) != 0) ? 1 : 0;
	num_count += (target_line_mask.at<uchar>(point.y + 1, point.x + 1) != 0) ? 1 : 0;

	return num_count;
}
cv::Point2d find_end_point(cv::Mat target_line_mask) {
	cv::Point2d point = cv::Point2d(-1, -1);

	for (int i = 0; i < target_line_mask.rows; i++) {
		for (int j = 0; j < target_line_mask.cols; j++) {
			if (target_line_mask.at<uchar>(i, j)) {
				point = cv::Point2d(j, i);
				//break;
			}
		}
		//if (point.x != -1 && point.y != -1)
			//break;
	}


	while (count_boundary_point(target_line_mask, point) != 1) {
		target_line_mask.at<uchar>(point.y, point.x) = 0;

		vector<cv::Point2d> neighbors = {
			cv::Point2d(point.x - 1, point.y),
			cv::Point2d(point.x + 1, point.y),
			cv::Point2d(point.x, point.y - 1),
			cv::Point2d(point.x, point.y + 1),
			cv::Point2d(point.x - 1, point.y - 1),
			cv::Point2d(point.x - 1, point.y + 1),
			cv::Point2d(point.x + 1, point.y - 1),
			cv::Point2d(point.x + 1, point.y + 1),
		};

		for (const auto& neighbor : neighbors) {
			if (target_line_mask.at<uchar>(neighbor.y, neighbor.x)) {
				point = neighbor;
			}
		}
	}
	return point;

}

vector<cv::Point2d> find_track_path(cv::Mat target_line_mask, cv::Point2d point) {
	vector<cv::Point2d> path;

	while (count_boundary_point(target_line_mask, point) != 1) {
		target_line_mask.at<uchar>(point.y, point.x) = 0;
		path.push_back(point);


		vector<cv::Point2d> neighbors = {
			cv::Point2d(point.x - 1, point.y),
			cv::Point2d(point.x + 1, point.y),
			cv::Point2d(point.x, point.y - 1),
			cv::Point2d(point.x, point.y + 1),
			cv::Point2d(point.x - 1, point.y - 1),
			cv::Point2d(point.x - 1, point.y + 1),
			cv::Point2d(point.x + 1, point.y - 1),
			cv::Point2d(point.x + 1, point.y + 1),
		};

		for (const auto& neighbor : neighbors) {
			if (target_line_mask.at<uchar>(neighbor.y, neighbor.x)) {
				point = neighbor;
			}
		}
	}
	path.push_back(point);

	return path;
}

vector<cv::Point2d> sort_points(const cv::Mat& target_line_mask) {
	cv::Mat mask_copy;
	cv::Mat mask_copy2;
	target_line_mask.copyTo(mask_copy);
	target_line_mask.copyTo(mask_copy2);


	cv::Point2d end_point = find_end_point(mask_copy);

	vector<cv::Point2d> result = find_track_path(mask_copy2, end_point);

	return result;
}

vector<double> hermite_spline(double x1, double y1, double g1, double x2, double y2, double g2) {
	vector<double> coefficients;

	cv::Mat A = (cv::Mat_<double>(4, 4) << pow(x1, 3), pow(x1, 2), x1, 1,
		pow(x2, 3), pow(x2, 2), x2, 1,
		3 * pow(x1, 2), 2 * x1, 1, 0,
		3 * pow(x2, 2), 2 * x2, 1, 0);

	cv::Mat B = (cv::Mat_<double>(4, 1) << y1, y2, g1, g2);

	cv::Mat solution;
	cv::solve(A, B, solution, cv::DECOMP_LU);

	cv::Mat coefficientsMat = solution.reshape(1);
	coefficients = vector<double>(coefficientsMat.begin<double>(), coefficientsMat.end<double>());

	return coefficients;
}

double substitute(vector<double> coefficients, double x) {
	return coefficients[0] * pow(x, 3) + coefficients[1] * pow(x, 2) + coefficients[2] * x + coefficients[3];
}

pair<vector<vector<double>>, vector<vector<double>>> fit(vector<double> x, vector<double> y, double k) {
	vector<vector<double>> poly_x;
	vector<vector<double>> poly_y;

	if (x.size() < 5) {
		return make_pair(poly_x, poly_y);
	}

	vector<double> xi = { x[0] };
	xi.insert(xi.end(), x.begin(), x.end());
	xi.push_back(x.back());

	vector<double> yi = { y[0] };
	yi.insert(yi.end(), y.begin(), y.end());
	yi.push_back(y.back());

	vector<double> gx, gy;

	for (int i = 1; i < xi.size() - 1; i++) {
		gx.push_back((xi[i + 1] - xi[i - 1]) / k);
		gy.push_back((yi[i + 1] - yi[i - 1]) / k);
	}

	for (int i = 0; i < x.size() - 1; i++) {
		vector<double> result_x = hermite_spline(0, x[i], gx[i], 1, x[i + 1], gx[i + 1]);
		vector<double> result_y = hermite_spline(0, y[i], gy[i], 1, y[i + 1], gy[i + 1]);

		poly_x.push_back(result_x);
		poly_y.push_back(result_y);
	}

	return make_pair(poly_x, poly_y);
}

vector<double> get_lines(vector<vector<double>> poly_x, int sample_num) {
	vector<double> result_x;
	double step = 1.0 / sample_num;

	for (int i = 0; i < poly_x.size(); i++) {
		for (int j = 0; j < sample_num; j++) {
			double x = substitute(poly_x[i], j * step);
			result_x.push_back(x);
		}
	}
	return result_x;
}

vector<vector<double>> differentiate(vector<vector<double>> poly_array) {
	vector<vector<double>> diff_array;

	for (auto poly : poly_array) {
		vector<double> diff_poly = { 0, 3 * poly[0], 2 * poly[1], poly[2] };
		diff_array.push_back(diff_poly);
	}
	return diff_array;
}

vector<double> simple_sampling(vector<double>& arr, int sparsity) {
	vector<double> sample_arr;
	for (int i = 0; i < arr.size(); i++) {
		if (i % sparsity == 0) {
			sample_arr.push_back(arr[i]);
		}
	}
	return sample_arr;
}

void delete_outliers(vector<double>& x_data, vector<double>& y_data, vector<double>& r_data, vector<double>& diff_data, double threshold) {
	double mean = accumulate(r_data.begin(), r_data.end(), 0.0) / r_data.size();
	double variance = accumulate(r_data.begin(), r_data.end(), 0.0, [mean](double acc, double x) { return acc + (x - mean) * (x - mean); }) / r_data.size();
	double std_dev = sqrt(variance);

	vector<double> r_copy = r_data;

	int count = 0;

	for (int i = 0; i < r_copy.size(); i++) {
		double score = (r_copy[i] - mean) / std_dev;
		if (abs(score) > threshold) {
			x_data.erase(x_data.begin() + (i - count));
			y_data.erase(y_data.begin() + (i - count));
			r_data.erase(r_data.begin() + (i - count));
			diff_data.erase(diff_data.begin() + (i - count));
			count++;
		}
	}
}