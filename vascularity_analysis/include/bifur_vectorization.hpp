#ifndef bifur_vectorization_hpp
#define bifur_vectorization_hpp

#include "opencv2/opencv.hpp"
#include <vector>

class Circle {

private:
    int MAX_R;
    std::vector<std::vector<std::pair<int, int>>> circle_edge_list;
    std::vector<std::vector<std::vector<int>>> circle_mask_list;

public:
    Circle(int max_r);

    double angle(double x_center, double y_center, double x, double y);

    //bool comparePair(std::pair<int, int>& p1,std::pair<int, int>& p2);

    std::vector<std::pair<int, int>> get_circle(int r);

    std::vector<std::vector<int>> get_circle_mask(int radius = 19);

    std::vector<std::vector<std::pair<int, int>>>& get_circle_edge_list();

    std::vector<std::vector<std::vector<int>>>& get_circle_mask_list();
    int get_MAX_R();
};

std::vector<int> get_pixel_values(cv::Mat& mask, std::vector<std::pair<int, int>>& coordinates);

void find_bifur_mask(cv::Mat& mask, int x, int y, Circle C, cv::Mat& output);

void draw_line(cv::Mat& mask, cv::Point point1, cv::Point point2, char color = 'r', int thickness = 2);
#endif 
