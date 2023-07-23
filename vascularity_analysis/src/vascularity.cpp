#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"
#include "vascularity.hpp"
#include "graph_structure.hpp"
#include "bifur_vectorization.hpp"
#include "points.hpp"
#include "skeletonize.hpp"

using namespace std;

vascularity::vascularity(cv::Mat img, cv::Mat vmask) {
	fundus = img;
	mask = vmask;

	make_graph();
}

void vascularity::make_graph() {
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