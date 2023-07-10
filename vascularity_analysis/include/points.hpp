#ifndef _points
#define _points

#include <vector>

#include "opencv2/opencv.hpp"
#include "points.hpp"

class points {
private:
	//bifur
	void get_X();
	void get_T();
	void get_Y();
	
	//end
	void get_E();

public:
	//bifur
	std::vector<cv::Mat> X;
	std::vector<cv::Mat> T;
	std::vector<cv::Mat> Y;

	//end
	std::vector<cv::Mat> E;
	
	//method
	points();

	
};

void test_points();

#endif
