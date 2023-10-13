#include <vector>
#include <iostream>
#include <queue>
#include <algorithm>
#include <cmath>
#include <numeric>

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
	cv::Mat fundus_channels[3];
	cv::Mat mask_for_skel[3];
	
	cv::split(mask, mask_channels);
	cv::split(fundus, fundus_channels);
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
	cv::Mat blur_fundus = fundus_channels[1];
	cv::GaussianBlur(blur_fundus, blur_fundus, cv::Size(5, 5), 0);

	for (int i = 0; i < 3; i++) {
		if (i == 1) continue;

		//connectedComponents
		cv::Mat labels, stats, cent;
		int retvals = cv::connectedComponentsWithStats(branch_skel[i], labels, stats, cent);
		
		//vectorization
		for (int j = 1; j < retvals; j++) {
			cv::Mat target_line = (labels == j);
			vector<cv::Point2d> sorted_points = sort_points(target_line);
			vbranch new_branch = get_branch_vector(sorted_points, mask_channels[i], blur_fundus);

			if (i == 0) v_graph.add_branch(new_branch);
			else if (i == 2) a_graph.add_branch(new_branch);
		}
	}
}

void vascularity::visualize(int sampling_dis) {
	//prepare data
	cv::Mat result = fundus.clone();
	cv::Mat result_split[3];
	cv::split(result, result_split);

	//visualze
	for (int i = 0; i < 3; i++) {
		vgraph target_graph;
		if (i == 1) continue;
		else if (i == 0) target_graph = v_graph;
		else if (i == 2) target_graph = a_graph;

		//bifur visualziation
		for (int j = 0; j < target_graph.get_bifur().size(); j++) {
			//bifur ���� ��������
			vbifur target_bifur = target_graph.get_bifur()[j];
			cv::Point center = target_bifur.get_center_coor();
			cv::Mat bifur_mask_seg = target_bifur.get_vbifur_mask();
			int circle_r = bifur_mask_seg.rows / 2;

			//�̹����� �߰�
			cv::Rect roi(center.x - circle_r, center.y - circle_r, bifur_mask_seg.rows, bifur_mask_seg.rows);
			cv::Mat target_patch = result_split[i](roi);
			target_patch = cv::max(target_patch, bifur_mask_seg);

			target_patch.copyTo(target_patch);
		}

		//branch visualization
		for (int j = 0; j < target_graph.get_branch().size(); j++) {
			//branch ���� ��������
			vbranch target_branch = target_graph.get_branch()[j];
			std::vector<std::vector<double>> poly_x = target_branch.get_poly_x();
			std::vector<std::vector<double>> poly_y = target_branch.get_poly_y();
			std::vector<std::vector<double>> poly_r = target_branch.get_poly_r();

			vector<double> x = get_lines(poly_x, sampling_dis);
			vector<double> y = get_lines(poly_y, sampling_dis);
			vector<double> r = get_lines(poly_r, sampling_dis);
			
			vector<vector<double>> diff_poly_x = differentiate(poly_x);
			vector<vector<double>> diff_poly_y = differentiate(poly_y);
			vector<double> diff_x = get_lines(diff_poly_x, sampling_dis);
			vector<double> diff_y = get_lines(diff_poly_y, sampling_dis);
			
			vector<double> center_tan;
			for (int k = 0; k < diff_x.size(); k++) 
				center_tan.push_back(diff_y[k] / (diff_x[k] + 1e-9));

			//delete_outliers(x, y, r, center_tan, 1);
			
			//�̹����� �߰�
			for (int k = 0; k < x.size(); k++) {
				int center_x = int(x[k]), center_y = int(y[k]);

				if (center_x > 0 && center_y > 0 && center_x < result.cols && center_y < result.rows) {
					result_split[1].at<uchar>(center_y, center_x) = 255;
					result_split[i].at<uchar>(center_y, center_x) = 255;
				}


				//side
				double angle = atan(center_tan[k]) + M_PI/2;
				int rt_x = int(x[k] + (r[k] * cos(angle))), lb_x = int(x[k] - (r[k] * cos(angle)));
				int rt_y = int(y[k] + (r[k] * sin(angle))), lb_y = int(y[k] - (r[k] * sin(angle)));

				if (rt_x > 0 && rt_y > 0 && rt_x < result.cols && rt_y < result.rows) 
					result_split[i].at<uchar>(rt_y, rt_x) = 255;

				if (lb_x > 0 && lb_y > 0 && lb_x < result.cols && lb_y < result.rows) 
					result_split[i].at<uchar>(lb_y, lb_x) = 255;

			}
		}
	}

	cv::merge(result_split, 3, result);

	cv::imwrite("vec.bmp", result);
	cv::imshow("result", result);
	cv::waitKey(0);
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

