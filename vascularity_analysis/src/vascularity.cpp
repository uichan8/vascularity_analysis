#include <vector>
#include <iostream>
#include <queue>

#include "opencv2/opencv.hpp"
#include "vascularity.hpp"
#include "graph_structure.hpp"
#include "bifur_vectorization.hpp"
#include "branch_vectorization.hpp"
#include "bifur_vectorization.hpp"
#include "skeletonize.hpp"

#define M_PI 3.14159265358979323846

using namespace std;

vascularity::vascularity(cv::Mat img, cv::Mat vmask) {
	fundus = img;
	mask = vmask;

	make_graph();
}


void vascularity::make_graph() {
	//체널 분리(의찬)
	cv::Mat mask_channels[3];
	cv::Mat mask_for_skel[3];
	cv::split(mask, mask_channels);
	cv::split(mask, mask_for_skel);

	//닫힘 연산(의찬) 스켈레톤에서 노이즈한 구멍들을 제거하기 위한 연산
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;
		dilate(mask_for_skel[i], mask_for_skel[i], element);
		erode(mask_for_skel[i], mask_for_skel[i], element);
	}

	//스켈레톤(의찬)
	cv::Mat skel_channels[3];
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;
		skeletonize(mask_for_skel[i], skel_channels[i]);
	}

	//bifur 위치 중심 마스크 계산하기(의찬)
	cv::Mat bifur_map[3];
	points P;
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;
		find_bifur_points(P, skel_channels[i], bifur_map[i]);
	}

	//bifur 중심 좌표 찾기(의찬)
	std::vector<cv::Point> bifur_coor[3];
	for (int i = 0; i < 3; i++)
		where(bifur_map[i], bifur_coor[i]);

	//팽창연산 길이가 1인 부분을 제대로 잡기 위한 연산
	for (int i = 0; i < 3; i++)
		dilate(mask_for_skel[i], mask_for_skel[i], element);

	//bifur 마스크 찾기, 유효 branch 중심 찾기(의찬)
	Circle C(19);
	vector<cv::Mat> bifur_mask[3];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < bifur_coor[i].size(); ++j) {
			cv::Mat bifur_mask_seg;
			vbifur new_bifur;
			int x = bifur_coor[i][j].x;
			int y = bifur_coor[i][j].y;
			new_bifur.set_center_coor(bifur_coor[i][j]);

			//bifur_mask 찾기
			find_bifur_mask(mask_channels[i], x, y, C, bifur_mask_seg);
			new_bifur.set_vbifur_mask(bifur_mask_seg);
			int circle_r = bifur_mask_seg.rows / 2;
			int circle_idx = circle_r / 2;
			vector<vector<int>> C_mask = C.get_circle_mask_list()[circle_idx];

			//마스크 정보 추가
			if (i == 0) v_graph.add_bifur(new_bifur);
			else if (i == 2) a_graph.add_bifur(new_bifur);
		}
	}

	// 그래프 구축
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;

		// 큐 생성 및 데이터 준비
		queue<vbifur> buffer_queue;
		cv::Mat skel_g = skel_channels[i].clone();

		/*바이퍼가 비어있을 때 예외처리 해줘야함*/

		if (i == 0) buffer_queue.push(v_graph.get_bifur()[0]);
		else if (i == 2) buffer_queue.push(a_graph.get_bifur()[0]);
		
		//큐가 비어버릴 때 까지 branch들을 추적
		while (!buffer_queue.empty()) {
			vbifur target_bifur = buffer_queue.front();
			buffer_queue.pop();
			cv::Point center = target_bifur.get_center_coor();
			vector<cv::Point2d> neighbors = {
				cv::Point2d(center.x + 1, center.y),
				cv::Point2d(center.x, center.y - 1),
				cv::Point2d(center.x, center.y + 1),
				cv::Point2d(center.x - 1, center.y),
				cv::Point2d(center.x - 1, center.y - 1),
				cv::Point2d(center.x - 1, center.y + 1),
				cv::Point2d(center.x + 1, center.y - 1),
				cv::Point2d(center.x + 1, center.y + 1)
			};
			
			vector<cv::Point> branch_start_point;
			for (int i = 0; i < 8; i++) {
				if (skel_g.at<uchar>(neighbors[i].y, neighbors[i].x))
					branch_start_point.push_back(neighbors[i]);
			}

			for (cv::Point start_point : branch_start_point) {
				vector<cv::Point2d> center_points;
				cv::Point2d end_branch;
				center_points = track_branch_centerline(start_point, skel_g, bifur_map[i], end_branch);
				//branch vector를 만들고
				vbranch new_branch;
				//todo

				//graph구조에 추가한다.
				vbifur end_bifur;
				if (end_branch.x == -1) {
					//빈 브랜치를 endpoint로 삽입
				}
				else {
					if (i == 0)
						end_bifur = v_graph.find_bifur(end_branch);
					else if (i == 2)
						end_bifur = a_graph.find_bifur(end_branch);
				}
				//브랜치랑 바이퍼를 연결시킨다.
			}
		}
	}
}


void vascularity::where(const cv::Mat& skel, std::vector<cv::Point> &result) {
	for (int y = 0; y < skel.rows; ++y) {
		for (int x = 0; x < skel.cols; ++x) {
			if (skel.at<uchar>(y, x) > 0) {
				result.emplace_back(x, y); 
			}
		}
	}
}