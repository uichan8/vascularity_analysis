#include <vector>
#include <iostream>
#include <queue>
#include <algorithm>

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

	simple_vectorization();
}

void vascularity::make_graph() {
	//ü�� �и�(����)
	cv::Mat mask_channels[3];
	cv::Mat mask_for_skel[3];
	cv::split(mask, mask_channels);
	cv::split(mask, mask_for_skel);

	//���� ����(����) ���̷��濡�� �������� ���۵��� �����ϱ� ���� ����
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;
		dilate(mask_for_skel[i], mask_for_skel[i], element);
		erode(mask_for_skel[i], mask_for_skel[i], element);
	}

	//���̷���(����)
	cv::Mat skel_channels[3];
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;
		skeletonize(mask_for_skel[i], skel_channels[i]);
	}

	//bifur ��ġ �߽� ����ũ ����ϱ�(����)
	cv::Mat bifur_map[3];
	points P;
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;
		find_bifur_points(P, skel_channels[i], bifur_map[i]);
	}

	//bifur �߽� ��ǥ ã��(����)
	std::vector<cv::Point> bifur_coor[3];
	for (int i = 0; i < 3; i++)
		where(bifur_map[i], bifur_coor[i]);

	//��â���� ���̰� 1�� �κ��� ����� ��� ���� ����
	for (int i = 0; i < 3; i++)
		dilate(mask_for_skel[i], mask_for_skel[i], element);

	//bifur ����ũ ã��, ��ȿ branch �߽� ã��(����)
	Circle C(19);
	vector<cv::Mat> bifur_mask[3];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < bifur_coor[i].size(); ++j) {
			cv::Mat bifur_mask_seg;
			vbifur new_bifur;
			int x = bifur_coor[i][j].x;
			int y = bifur_coor[i][j].y;
			new_bifur.set_center_coor(bifur_coor[i][j]);

			//bifur_mask ã��
			find_bifur_mask(mask_channels[i], x, y, C, bifur_mask_seg);
			new_bifur.set_vbifur_mask(bifur_mask_seg);

			//����ũ ���� �߰�
			if (i == 0) v_graph.add_bifur(new_bifur);
			else if (i == 2) a_graph.add_bifur(new_bifur);
		}
	}

	// �׷��� ����
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;

		// ť ���� �� ������ �غ�
		queue<vbifur> buffer_queue;
		cv::Mat skel_g = skel_channels[i].clone();

		/*�����۰� ������� �� ����ó�� �������*/

		if (i == 0) buffer_queue.push(v_graph.get_bifur()[0]);
		else if (i == 2) buffer_queue.push(a_graph.get_bifur()[0]);
		
		//ť�� ������ �� ���� branch���� ����
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
			
			queue<cv::Point> branch_start_point;
			for (int i = 0; i < 8; i++) {
				if (skel_g.at<uchar>(neighbors[i].y, neighbors[i].x))
					branch_start_point.push(neighbors[i]);
			}

			/*
			for (cv::Point start_point : branch_start_point) {
				vector<cv::Point2d> center_points;
				cv::Point2d end_branch;
				center_points = track_branch_centerline(start_point, skel_g, bifur_map[i], end_branch);
				//branch vector�� �����
				vbranch new_branch;
				//todo

				//graph������ �߰��Ѵ�.
				vbifur end_bifur;
				if (end_branch.x == -1) {
					//�� �귣ġ�� endpoint�� ����
				}
				else {
					if (i == 0)
						end_bifur = v_graph.find_bifur(end_branch);
					else if (i == 2)
						end_bifur = a_graph.find_bifur(end_branch);
				}
				//�귣ġ�� �����۸� �����Ų��.
			}
			*/
		}
	}
}

void vascularity::simple_vectorization() {
	//ü�� �и�
	cv::Mat mask_channels[3];
	cv::Mat mask_for_skel[3];
	cv::split(mask, mask_channels);
	cv::split(mask, mask_for_skel);

	//���� ���� ���̷��濡�� �������� ���۵��� �����ϱ� ���� ����
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;
		dilate(mask_for_skel[i], mask_for_skel[i], element);
		erode(mask_for_skel[i], mask_for_skel[i], element);
	}

	//���̷���
	cv::Mat skel_channels[3];
	for (int i = 0; i < 3; i++) {
		skeletonize(mask_for_skel[i], skel_channels[i]);
	}

	//bifur ��ġ �߽� ����ũ ����ϱ�
	cv::Mat bifur_map[3];
	points P;
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;
		find_bifur_points(P, skel_channels[i], bifur_map[i]);
	}

	//bifur �߽� ��ǥ ã��
	std::vector<cv::Point> bifur_coor[3];
	for (int i = 0; i < 3; i++)
		where(bifur_map[i], bifur_coor[i]);

	//��â���� ���̰� 1�� �κ��� ����� ��� ���� ����
	for (int i = 0; i < 3; i++)
		dilate(mask_for_skel[i], mask_for_skel[i], element);

	//bifur vectorization
	Circle C(19);
	cv::Mat bifur_skel[3];
	for (int i = 0; i < 3; i++) {
		//bifur_skel �ʱ�ȭ
		bifur_skel[i] = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
		for (int j = 0; j < bifur_coor[i].size(); ++j) {
			//bifur center ���� �Է�
			vbifur new_bifur;
			int x = bifur_coor[i][j].x;
			int y = bifur_coor[i][j].y;
			new_bifur.set_center_coor(bifur_coor[i][j]);

			//bifur_mask ���� �Է�
			cv::Mat bifur_mask_seg;
			find_bifur_mask(mask_channels[i], x, y, C, bifur_mask_seg);
			new_bifur.set_vbifur_mask(bifur_mask_seg);

			//�׷����� ����ũ ���� �߰�
			if (i == 0) v_graph.add_bifur(new_bifur);
			else if (i == 2) a_graph.add_bifur(new_bifur);

			//branch_�� �ش��ϴ� skel�κ� ���
			int circle_r = bifur_mask_seg.rows / 2;
			int circle_idx = circle_r / 2;
			cv::Rect roi(x - circle_r, y - circle_r, bifur_mask_seg.rows, bifur_mask_seg.rows);

			cv::Mat C_mask = C.get_circle_mask_list()[circle_idx];
			cv::Mat bifur_skel_seg = skel_channels[i](roi);
			
			bifur_skel_seg.copyTo(bifur_skel[i](roi));
		}
	}
	
	// ���̷��濡�� ������ �κ� ����
	cv::Mat branch_skel[3];
	for (int i = 0; i < 3; i++) {
		branch_skel[i] = skel_channels[i].clone();
		cv::subtract(branch_skel[i], bifur_skel[i], branch_skel[i]);
	}

	//branch vectorization
	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;

		//connectedComponents
		cv::Mat labels, stats, cent;
		int retvals = cv::connectedComponentsWithStats(branch_skel[i], labels, stats, cent);
		
		//vectorization
		for (int j = 1; j < retvals; j++) {
			cv::Mat target_line = (labels == j);
			vector<cv::Point2d> sorted_points = sort_points(target_line);
			vbranch new_branch = get_branch_vector(sorted_points, mask, fundus);

			if (i == 0) v_graph.add_branch(new_branch);
			else if (i == 2) a_graph.add_branch(new_branch);
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