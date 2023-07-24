#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "vascularity.hpp"
#include "graph_structure.hpp"
#include "bifur_vectorization.hpp"
#include "points.hpp"
#include "branch_vectorization.hpp"
#include "vessel_width_detection.hpp"
#include "bifur_vectorization.hpp"
#define M_PI 3.14159265358979323846
#include "skeletonize.hpp"

using namespace std;

vascularity::vascularity(cv::Mat img, cv::Mat vmask) {
	fundus = img;
	mask = vmask;

	make_graph();
}

void vascularity::make_graph(){
	//체널 분리
	cv::Mat branch_mask;
	mask.copyTo(branch_mask);

	cv::Mat blur_fundus;
	cv::GaussianBlur(fundus, blur_fundus, cv::Size(5, 5), 3);

	//체널 분리(의찬)
	cv::Mat mask_channels[3];
	cv::split(mask, mask_channels);

	//스켈레톤(의찬)
	cv::Mat skel_channels[3];
	for (int i = 0; i < 3; i++)
		skeletonize(mask_channels[i], skel_channels[i]);

	//bifur 위치 중심 마스크 계산하기(의찬)
	cv::Mat bifur_map[3];
	points P;
	for (int i = 0; i < 3; i++)
		P.find_bifur_points(skel_channels[i], bifur_map[i]);

	//bifur 중심 좌표 찾기(의찬)
	std::vector<cv::Point> bifur_coor[3];
	for (int i = 0; i < 3; i++)
		where(bifur_map[i], bifur_coor[i]);

	//bifur 마스크 찾기, 유효 branch 중심 찾기(의찬)
	Circle C(19);
	vector<cv::Mat> bifur_mask[3];
	cv::Mat branch_map[3];
	for (int i = 0; i < 3; i++)
		branch_map[i] = skel_channels[i].clone();

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < bifur_coor[i].size(); ++j) {
			cv::Mat bifur_mask_seg;
			int x = bifur_coor[i][j].x;
			int y = bifur_coor[i][j].y;
			find_bifur_mask(mask_channels[i], x, y, C, bifur_mask_seg);
			int circle_r = bifur_mask_seg.rows / 2;
			int circle_idx = circle_r / 2;
			vector<vector<int>> C_mask = C.get_circle_mask_list()[circle_idx];
			for (int k = 0; k < bifur_mask_seg.rows; k++) {
				for (int l = 0; l < bifur_mask_seg.rows; l++) {
					if (C_mask[k][l]) {
						int target_x = x + l - circle_r, target_y = y + k - circle_r;
						branch_map[i].at<uchar>(target_y, target_x) = 0;
					}
				}
			}
		}
	}

	//vertex 분리 (종수)
	cv::Mat labels_r, stats_r, cent_r, labels_b, stats_b, cent_b;
	int retvals_r = cv::connectedComponentsWithStats(branch_map[2], labels_r, stats_r, cent_r);
	int retvals_b = cv::connectedComponentsWithStats(branch_map[0], labels_b, stats_b, cent_b);

	//vessel_informs (종수)
	vector<tuple<int, cv::Mat, cv::Mat, char>> vessel_informs = {
		make_tuple(retvals_r, labels_r, mask_channels[2], 'r'),
		make_tuple(retvals_b, labels_b, mask_channels[0], 'b')
	};
    
	vector<cv::Point2d> center, sub_edge;
	vector<double> r;

	// branch_vectorization(종수)
	for (auto& vessel : vessel_informs) {
		sub_edge.clear();
		
		int retvals = get<0>(vessel);
		cv::Mat labels = get<1>(vessel);
		cv::Mat bmask = get<2>(vessel);
		char color = get<3>(vessel);
		//각각의 branch 호출해서 vectorization
		for (int i = 1; i < retvals; i++) {
			cv::Mat target_line = (labels == i);
			vector<cv::Point2d> sorted_points = sort_points(target_line);

			//마스크 맵 기반 혈관 굵기 추정
			tuple<vector<double>, vector<double>, vector<double>, vector<double>> edge = mask_witdth_detection(bmask, sorted_points);

			vector<double> edge_x = get<0>(edge);
			vector<double> edge_x2 = get<1>(edge);
			vector<double> edge_y = get<2>(edge);
			vector<double> edge_y2 = get<3>(edge);

			if (get<0>(edge).size() > 4) {
				draw_line(branch_mask, cv::Point2d(edge_y[1], edge_x[1]), cv::Point2d(edge_y2[1], edge_x2[1]), color);
				draw_line(branch_mask, cv::Point2d(edge_y[edge_y.size() - 2], edge_x[edge_x.size() - 2]), cv::Point2d(edge_y2[edge_y2.size() - 2], edge_x2[edge_x2.size() - 2]), color);
			}

            //포인트 샘플링
			edge_x = simple_sampling(edge_x, 2);
			edge_y = simple_sampling(edge_y, 2);
			edge_x2 = simple_sampling(edge_x2, 2);
			edge_y2 = simple_sampling(edge_y2, 2);

			vector<double> x_cen(edge_x.size());
			vector<double> y_cen(edge_x.size());
			vector<double> center_tan(edge_x.size());
			vector<double> vessel_w(edge_x.size());

			for (size_t i = 0; i < edge_x.size(); i++) {
				x_cen[i] = (edge_x[i] + edge_x2[i]) / 2.0;
				y_cen[i] = (edge_y[i] + edge_y2[i]) / 2.0;
				center_tan[i] = (edge_y[i] - edge_y2[i]) / (edge_x[i] - edge_x2[i] + 1e-12);
				vessel_w[i] = sqrt(pow((edge_y[i] - edge_y2[i]), 2) + pow((edge_x[i] - edge_x2[i]), 2)) / 2.0;
			}
			// subpixel localization
			for (size_t i = 0; i < x_cen.size(); i++) {
				vector<cv::Point2d> edge_coor;
				edge_coor = get_edge(blur_fundus, cv::Point2d(x_cen[i], y_cen[i]), center_tan[i], vessel_w[i]);
				x_cen[i] = (edge_coor[0].x + edge_coor[1].x) / 2;
				y_cen[i] = (edge_coor[0].y + edge_coor[1].y) / 2;
				//center.push_back(cv::Point2d((edge_coor[0].x + edge_coor[1].x) / 2, (edge_coor[0].y + edge_coor[1].y) / 2));
				r.push_back(sqrt(pow((edge_coor[0].x - edge_coor[1].x), 2) + pow((edge_coor[0].y - edge_coor[1].y), 2)) / 2);
			}

			int sampling_num = 1;
			vector<vector<double>> spline_diff_x, spline_diff_y, spline_diff_poly;
			vector<double> spline_diff, r_len, angle;

			pair<vector<vector<double>>, vector<vector<double>>> spline = fit(x_cen, y_cen, 1.5);

			// supixel_localization point 기반 혈관 중심과 edge_point 개선
			x_cen = get_lines(spline.first, sampling_num);
			y_cen = get_lines(spline.second, sampling_num);
			spline_diff_x = differentiate(spline.first);
			spline_diff_y = differentiate(spline.second);

			for (size_t i = 0; i < spline_diff_x.size(); i++) {
				vector<double> spline_diff_poly_row;
				for (size_t j = 0; j < spline_diff_x[i].size(); j++) {
					spline_diff_poly_row.push_back(spline_diff_y[i][j] / (spline_diff_x[i][j] + 1e-9));
				}
				spline_diff_poly.push_back(spline_diff_poly_row);
			}
			spline_diff = get_lines(spline_diff_poly, sampling_num);

			for (int i = 0; i < r.size(); i++) {
				r_len.push_back(static_cast<double>(i));
			}
			pair<vector<vector<double>>, vector<vector<double>>> spline_r = fit(r_len, r, 1.5);
			
			r = get_lines(spline_r.second, sampling_num);

			delete_outliers(x_cen, y_cen, r, spline_diff, 2);

			for (const auto& diff : spline_diff) {
				angle.push_back(atan(diff) + M_PI / 2);
			}

			for (size_t i = 0; i < x_cen.size(); i++) {
				sub_edge.push_back(cv::Point2d(x_cen[i] + r[i] * cos(angle[i]), y_cen[i] + r[i] * sin(angle[i])));
				sub_edge.push_back(cv::Point2d(x_cen[i] + r[i] * cos(angle[i] + M_PI), y_cen[i] + r[i] * sin(angle[i]) + M_PI));
				center.push_back(cv::Point2d(x_cen[i], y_cen[i]));
			}
		}
	}

	
}



void vascularity::where(const cv::Mat& skel, std::vector<cv::Point> &result) {
	for (int y = 0; y < skel.rows; ++y) {
		for (int x = 0; x < skel.cols; ++x) {
			if (skel.at<uchar>(y, x) > 0) {
				result.emplace_back(x, y); //작동이 이상할 경우 push back으로 바꿔볼것
			}
		}
	}
}