#ifndef _points
#define _points

#include <vector>

#include "opencv2/opencv.hpp"

class points {
private:
	//bifur
	std::vector<cv::Mat> X;
	std::vector<cv::Mat> T;
	std::vector<cv::Mat> Y;
	
	//end
	std::vector<cv::Mat> E;
	
public:
	//method
	points();
	void find_bifur_points(const cv::Mat &skel_mask, cv::Mat& result_point_mask);
	void find_end_points(const cv::Mat& skel_mask, cv::Mat& result_point_mask);
};

#endif

