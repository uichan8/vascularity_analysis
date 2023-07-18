#include <vector>

#include "opencv2/opencv.hpp"
#include "points.hpp"

using namespace std;

points::points() {
	cv::Mat X0 = (cv::Mat_<int>(3, 3) << -1, 1, -1, 1, 1, 1, -1, 1, -1);
	cv::Mat X1 = (cv::Mat_<int>(3, 3) << 1, -1, 1, -1, 1, -1, 1, -1, 1);
	X.insert(X.end(), {X0,X1});
	
	cv::Mat T0 = (cv::Mat_<int>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 0, 0);
	cv::Mat T1 = (cv::Mat_<int>(3, 3) << 1, 0, 1, 0, 1, 0, 1, 0, 0);
	cv::Mat T2 = (cv::Mat_<int>(3, 3) << 0, 1, 0, 1, 1, 0, 0, 1, 0);
	cv::Mat T3 = (cv::Mat_<int>(3, 3) << 1, 0, 0, 0, 1, 0, 1, 0, 1);
	cv::Mat T4 = (cv::Mat_<int>(3, 3) << 0, 0, 0, 1, 1, 1, 0, 1, 0);
	cv::Mat T5 = (cv::Mat_<int>(3, 3) << 0, 0, 1, 0, 1, 0, 1, 0, 1);
	cv::Mat T6 = (cv::Mat_<int>(3, 3) << 0, 1, 0, 0, 1, 1, 0, 1, 0);
	cv::Mat T7 = (cv::Mat_<int>(3, 3) << 1, 0, 1, 0, 1, 0, 0, 0, 1);
	T.insert(T.end(), { T0,T1,T2,T3,T4,T5,T6,T7 });

	cv::Mat Y0 = (cv::Mat_<int>(3, 3) << 1, -1, 1, -1, 1, -1, 0, 1, 0);
	cv::Mat	Y1 = (cv::Mat_<int>(3, 3) << -1, 1, -1, 1, 1, 0, -1, 0, 1);
	cv::Mat Y2 = (cv::Mat_<int>(3, 3) << 1, -1, 0, 1, 1, 1, 1, -1, 0);
	cv::Mat Y3 = (cv::Mat_<int>(3, 3) << -1, 0, 1, 1, 1, 0, -1, 1, -1);
	cv::Mat Y4 = (cv::Mat_<int>(3, 3) << 0, 1, 0, -1, 1, -1, 1, -1, 1);
	cv::Mat Y5 = (cv::Mat_<int>(3, 3) << 1, 0, -1, 0, 1, 1, -1, 1, -1);
	cv::Mat Y6 = (cv::Mat_<int>(3, 3) << 0, -1, 1, 1, 1, -1, 0, -1, 1);
	cv::Mat Y7 = (cv::Mat_<int>(3, 3) << -1, 1, -1, 0, 1, 1, 1, 0, -1);
	Y.insert(Y.end(), { Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7 });

	cv::Mat E0 = (cv::Mat_<int>(3, 3) << -1, -1, -1, -1, 1, -1, 0, 1, 0);
	cv::Mat E1 = (cv::Mat_<int>(3, 3) << -1, -1, -1, -1, 1, 0, -1, 0, 1);
	cv::Mat E2 = (cv::Mat_<int>(3, 3) << -1, -1, 0, -1, 1, 1, -1, -1, 0);
	cv::Mat E3 = (cv::Mat_<int>(3, 3) << -1, 0, 1, -1, 1, 0, -1, -1, -1);
	cv::Mat E4 = (cv::Mat_<int>(3, 3) << 0, 1, 0, -1, 1, -1, -1, -1, -1);
	cv::Mat E5 = (cv::Mat_<int>(3, 3) << 1, 0, -1, 0, 1, -1, -1, -1, -1);
	cv::Mat E6 = (cv::Mat_<int>(3, 3) << 0, -1, -1, 1, 1, -1, 0, -1, -1);
	cv::Mat E7 = (cv::Mat_<int>(3, 3) << -1, -1, -1, 0, 1, -1, 1, 0, -1);
	E.insert(E.end(), { E0,E1,E2,E3,E4,E5,E6,E7 });
}

void points::find_bifur_points(cv::Mat& skel_mask, cv::Mat& result_point_mask){
	cv::Mat output_img = cv::Mat::zeros(skel_mask.size(), skel_mask.type());
	cv::Mat kerneloutput = cv::Mat::zeros(skel_mask.size(), skel_mask.type());

	//X bifur
	for (int i = 0; i < X.size(); i++){
		cv::morphologyEx(skel_mask, kerneloutput, cv::MORPH_HITMISS, X[i]);
		output_img = output_img + kerneloutput;
	}

	// T bifur
	for (int i = 0; i < T.size(); i++){
		cv::morphologyEx(skel_mask, kerneloutput, cv::MORPH_HITMISS, T[i]);
		output_img = output_img + kerneloutput;
	}

	// Y bifur
	for (int i = 0; i < Y.size(); i++){
		cv::morphologyEx(skel_mask, kerneloutput, cv::MORPH_HITMISS, Y[i]);
		output_img = output_img + kerneloutput;
	}

	// return result
	result_point_mask = output_img;
}

void points::find_end_points(cv::Mat& skel_mask, cv::Mat& result_point_mask) {
	cv::Mat output_img = cv::Mat::zeros(skel_mask.size(), skel_mask.type());
	cv::Mat kerneloutput = cv::Mat::zeros(skel_mask.size(), skel_mask.type());

	//E end
	for (int i = 0; i < E.size(); i++) {
		cv::morphologyEx(skel_mask, kerneloutput, cv::MORPH_HITMISS, E[i]);
		output_img = output_img + kerneloutput;
	}

	// return result
	result_point_mask = output_img;
}

