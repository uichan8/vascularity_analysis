#ifndef _skeletonize
#define _skeletonize

#include "opencv2/opencv.hpp"
#include <vector>

void skeletonize(const cv::Mat& mask, cv::Mat& skel);
void compute_thin_image(cv::Mat& img);
void find_simple_point_candidates(cv::Mat& img, int curr_border, std::vector<cv::Point3d>& simple_border_points, std::vector<int>& LUT);

void get_neighborhood(cv::Mat& img, int p, int r, int c, int* neighborhood);

bool is_endpoint(int* neighborhood);
bool is_Euler_invariant(int* neighborhood, std::vector<int>& LUT);
bool is_simple_point(int* neighborhood);

//LOOK UP TABLE
void fill_Euler_LUT(std::vector<int>& LUT);
void octree_labeling(int octant, int label, int *cube);
#endif
