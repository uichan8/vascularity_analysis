#include <vector>

#include "opencv2/opencv.hpp"
#include "vascularity.hpp"
#include "graph_structure.hpp"
#include "points.hpp"
#include "branch_vectorization.hpp"
#include "vessel_width_detection.hpp"
#include "bifur_vectorization.hpp"

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

	cv::Mat mask_channels[3];
	cv::split(mask, mask_channels);

	//스켈레톤
	cv::Mat skel_channels[3];
	for (int i = 0; i < 3; i++)
		skeletonize(mask_channels[i], skel_channels[i]);

	//스플릿
	cv::Mat branch_map[3], bifur_map[3];
	for (int i = 0; i < 3; i++)
		branch_mask_split(skel_channels[i], branch_map[i], bifur_map[i]);

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

	for (auto& vessel : vessel_informs) {
		sub_edge.clear();
		
		int retvals = get<0>(vessel);
		cv::Mat labels = get<1>(vessel);
		cv::Mat bmask = get<2>(vessel);
		char color = get<3>(vessel);

		for (int i = 1; i < retvals; i++) {
			cv::Mat target_line = (labels == i);
			vector<cv::Point2d> sorted_points = sort_points(target_line);

			tuple<vector<double>, vector<double>, vector<double>, vector<double>> edge = mask_witdth_detection(bmask, sorted_points);

			vector<double> edge_x = get<0>(edge);
			vector<double> edge_x2 = get<1>(edge);
			vector<double> edge_y = get<2>(edge);
			vector<double> edge_y2 = get<3>(edge);

			if (get<0>(edge).size() > 4) {
				draw_line(branch_mask, cv::Point2d(edge_y[1], edge_x[1]), cv::Point2d(edge_y2[1], edge_x2[1]), color);
				draw_line(branch_mask, cv::Point2d(edge_y[edge_y.size() - 2], edge_x[edge_x.size() - 2]), cv::Point2d(edge_y2[edge_y2.size() - 2], edge_x2[edge_x2.size() - 2]), color);
			}
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

			for (size_t i = 0; i < x_cen.size(); i++) {
				vector<cv::Point2d> edge_coor;
				edge_coor = get_edge(blur_fundus, cv::Point2d(x_cen[i], y_cen[i]), center_tan[i], vessel_w[i]);
				x_cen[i] = (edge_coor[0].x + edge_coor[1].x) / 2;
				y_cen[i] = (edge_coor[0].y + edge_coor[1].y) / 2;
				//center.push_back(cv::Point2d((edge_coor[0].x + edge_coor[1].x) / 2, (edge_coor[0].y + edge_coor[1].y) / 2));
				r.push_back(sqrt(pow((edge_coor[0].x - edge_coor[1].x), 2) + pow((edge_coor[0].y - edge_coor[1].y), 2)) / 2);
			}

			int sampling_num = 1;

			pair<vector<vector<double>>, vector<vector<double>>> spline = fit(x_cen, y_cen, 1.5);

			x_cen = get_lines(spline.first, sampling_num);
			y_cen = get_lines(spline.second, sampling_num);
            


		}
	}



	cv::imshow("Result", branch_map[0]);
	cv::waitKey(0);
}

void vascularity::skeletonize(const cv::Mat& src, cv::Mat& dst) {
	dst = src.clone();
	dst /= 255;         // convert to binary image

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	do {
		skel_iteration(dst, 0);
		skel_iteration(dst, 1);
		cv::absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	dst *= 255;
}

void vascularity::skel_iteration(cv::Mat& img, int iter){
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar* pAbove;
	uchar* pCurr;
	uchar* pBelow;
	uchar* nw, * no, * ne;    // north (pAbove)
	uchar* we, * me, * ea;
	uchar* sw, * so, * se;    // south (pBelow)

	uchar* pDst;

	// initialize row pointers
	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);

	for (y = 1; y < img.rows - 1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = img.ptr<uchar>(y + 1);

		pDst = marker.ptr<uchar>(y);

		// initialize col pointers
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x) {
			// shift col pointers left by one (scan left to right)
			nw = no;
			no = ne;
			ne = &(pAbove[x + 1]);
			we = me;
			me = ea;
			ea = &(pCurr[x + 1]);
			sw = so;
			so = se;
			se = &(pBelow[x + 1]);

			int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
				(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
				(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
				(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
			int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
			int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
			int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				pDst[x] = 1;
		}
	}

	img &= ~marker;
}

void vascularity::branch_mask_split(const cv::Mat& skel_mask, cv::Mat& branch_map, cv::Mat& bifur_map) {
	points P;
	P.find_bifur_points(skel_mask, bifur_map); //코드 효율성에 따라 밖으로 뺄 수 도 있음

	cv::Mat dilatedImage;
	cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::dilate(bifur_map, dilatedImage, dilateKernel);

	cv::subtract(skel_mask, dilatedImage, branch_map);
}