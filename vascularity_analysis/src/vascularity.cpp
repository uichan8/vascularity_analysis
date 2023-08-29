#include <vector>
#include <iostream>
#include <queue>

#include "opencv2/opencv.hpp"
#include "vascularity.hpp"
#include "graph_structure.hpp"
#include "bifur_vectorization.hpp"
#include "points.hpp"
#include "branch_vectorization.hpp"
#include "vessel_width_detection.hpp"
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
	//체널 분리
	cv::Mat branch_mask;
	mask.copyTo(branch_mask);

	cv::Mat blur_fundus;
	cv::GaussianBlur(fundus, blur_fundus, cv::Size(5, 5), 3);
	cv::cvtColor(blur_fundus, blur_fundus, cv::COLOR_BGR2GRAY);

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
	for (int i = 0; i < 3; i++){
		if (i == 1) continue;
		P.find_bifur_points(skel_channels[i], bifur_map[i]);
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
	cv::Mat branch_map[3];
	for (int i = 0; i < 3; i++) {
		branch_map[i] = mask_for_skel[i].clone();
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
			if (i == 2) a_graph.add_bifur(new_bifur);
			else if (i == 0) v_graph.add_bifur(new_bifur);

			//유효 branch 계산
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
	vector<cv::Point2d> sub;

	// branch_vectorization(종수)
	for (auto& vessel : vessel_informs) {
		//sub_edge.clear();

		int retvals = get<0>(vessel);
		cv::Mat labels = get<1>(vessel);
		cv::Mat bmask = get<2>(vessel);
		char color = get<3>(vessel);

		//각각의 branch 호출해서 vectorization
		for (int i = 1; i < retvals; i++) {
			cv::Mat target_line = (labels == i);
			vector<cv::Point2d> sorted_points = sort_points(target_line);

			if (sorted_points.size() < 6) // -> 이 부분에서 너무 짧은 branch는 버리는게 아니라 마스크 기반으로 그냥 냅드는 게 나은 듯
				continue;

			//양 끝점 정보 입력(의찬)
			vbranch new_branch;
			new_branch.set_end_points(sorted_points[0], sorted_points.back());

			//마스크 맵 기반 혈관 굵기 추정
			tuple<vector<double>, vector<double>, vector<double>, vector<double>> edge = mask_witdth_detection(bmask, sorted_points);

			vector<double> edge_x = get<0>(edge);
			vector<double> edge_x2 = get<1>(edge);
			vector<double> edge_y = get<2>(edge);
			vector<double> edge_y2 = get<3>(edge);

			//bifur를 찾기 위한 mask
			if (edge_x.size() > 4) {
				draw_line(branch_mask, cv::Point2d(edge_x[1], edge_y[1]), cv::Point2d(edge_x2[1], edge_y2[1]), color);
				draw_line(branch_mask, cv::Point2d(edge_x[edge_x.size() - 2], edge_y[edge_y.size() - 2]), cv::Point2d(edge_x2[edge_x2.size() - 2], edge_y2[edge_y2.size() - 2]), color);
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
			vector<double> r;

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

				sub.push_back(edge_coor[0]);
				sub.push_back(edge_coor[1]);

				x_cen[i] = (edge_coor[0].x + edge_coor[1].x) / 2;
				y_cen[i] = (edge_coor[0].y + edge_coor[1].y) / 2;
				r.push_back(sqrt(pow((edge_coor[0].x - edge_coor[1].x), 2) + pow((edge_coor[0].y - edge_coor[1].y), 2)) / 2);
			}

			int sampling_num = 1;
			vector<vector<double>> spline_diff_x, spline_diff_y, spline_diff_poly;
			vector<double> spline_diff, r_len, angle;

			pair<vector<vector<double>>, vector<vector<double>>> spline = fit(x_cen, y_cen, 1.5);

			// supixel_localization point 기반 혈관 중심과 edge_point 개선
			spline_diff_x = differentiate(spline.first);
			spline_diff_y = differentiate(spline.second);

			for (size_t i = 0; i < spline_diff_x.size(); i++) {
				vector<double> spline_diff_poly_row;
				for (size_t j = 0; j < spline_diff_x[i].size(); j++) {
					spline_diff_poly_row.push_back(spline_diff_y[i][j] / (spline_diff_x[i][j] + 1e-9));
				}
				spline_diff_poly.push_back(spline_diff_poly_row);
			}
			// branch center정보 추가 (의찬)
			new_branch.set_poly_x(spline.first);
			new_branch.set_poly_y(spline.second);

			spline_diff = get_lines(spline_diff_poly, sampling_num);

			for (int i = 0; i < r.size(); i++) {
				r_len.push_back(static_cast<double>(i));
			}
			pair<vector<vector<double>>, vector<vector<double>>> spline_r = fit(r_len, r, 1.5);

			//branch r 정보 추가 (의찬)
			new_branch.set_poly_r(spline_r.second);

			//그래프에 데이터 추가
			if (color == 'r') a_graph.add_branch(new_branch);
			else if (color == 'b') v_graph.add_branch(new_branch);
		}
		cv::imwrite("cut.bmp", branch_mask);
		cv::waitKey(0);
	}

	cv::Mat branch_channels[3];
	cv::split(branch_mask, branch_channels);

	retvals_r = cv::connectedComponentsWithStats(branch_channels[2], labels_r, stats_r, cent_r);
	retvals_b = cv::connectedComponentsWithStats(branch_channels[0], labels_b, stats_b, cent_b);

	cv::Mat mask_edge = cv::Mat::zeros(mask.size(), mask.type());

	cv::Mat edge_channels[3];
	cv::split(mask_edge, edge_channels);

	//Bifur 검출(종수)
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;
		std::vector<cv::Point> bc = bifur_coor[i];
		cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
		kernel.at<uchar>(0, 0) = 0;
		kernel.at<uchar>(0, 2) = 0;
		kernel.at<uchar>(2, 0) = 0;
		kernel.at<uchar>(2, 2) = 0;

		cv::Mat target_label;
		if (i == 0) target_label = labels_b;
		else target_label = labels_r;

		for (auto pts : bc) {
			int label = target_label.at<int>(static_cast<int>(pts.y), static_cast<int>(pts.x));
			if (cv::countNonZero(target_label == label) < 2000) {
				cv::Mat branch_edge = (target_label == label) * 255;
				cv::Mat dilated_mask;
				cv::dilate(branch_edge, dilated_mask, kernel, cv::Point(-1, -1), 1);
				cv::imshow("a", dilated_mask);
				cv::waitKey(0);
				dilated_mask &= (branch_channels[i] == 0);

				// bifur edge 정보 추가
				vector<cv::Point> edges;
				where(dilated_mask, edges);
				if (i == 0) v_graph.find_bifur(pts).set_bifur_edge(edges);
				else if (i == 2) a_graph.find_bifur(pts).set_bifur_edge(edges);
			}
		}
	}
	a_graph.connect();
	v_graph.connect();
	cv::imshow("a", branch_mask);
	cv::waitKey(0);
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
		P.find_bifur_points(skel_channels[i], bifur_map[i]);
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