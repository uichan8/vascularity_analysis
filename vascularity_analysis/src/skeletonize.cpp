#include <vector>
#include <string>

#include "skeletonize.hpp"
#include "opencv2/opencv.hpp"


using namespace std;

void skeletonize(const cv::Mat& mask, cv::Mat& skel) {
	//�����߰�
	int rows = mask.rows;
	int cols = mask.cols;

	//preprocess pad
	mask.convertTo(skel, CV_8SC1);
	cv::copyMakeBorder(skel, skel, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::Mat zero_c_pad = cv::Mat::zeros(rows + 2, cols + 2, CV_8SC3);

	int channels = zero_c_pad.channels();
	int rows2 = zero_c_pad.rows;
	int cols2 = zero_c_pad.cols;
	int step = zero_c_pad.step1();
	uchar* channelData = zero_c_pad.data + 1;

	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			channelData[r * step + c * channels] = skel.at<uchar>(r, c)>=1;
		}
	}
	skel = zero_c_pad;


	// cv::Mat zero_c_pad = cv::Mat::zeros(rows + 2, cols + 2, CV_8SC1);
	// std::vector<cv::Mat> split_c = { zero_c_pad , skel, zero_c_pad };
	// cv::merge(split_c, skel);

	//preprocess normalize
	//skel.setTo(1, skel != 0);

	//thinning
	compute_thin_image(skel);


	//����
	//ũ���ϰ� ���� �����ؾߵ�

	skel *= 255;
	cv::imwrite("skel_result001.bmp", skel);
}

void compute_thin_image(cv::Mat& img) {
	int unchanged_borders = 0, curr_border, num_borders = 4;
	int borders[6] = { 4, 3, 2, 1, 5, 6 };
	int p, r, c;
	bool no_change;

	cv::Point3d point; //p,r,c ����
	vector<cv::Point3d> simple_border_points;

	int num_border_points;
	int neighb[27];

	vector<int> LUT;
	fill_Euler_LUT(LUT);
	int cnt = 0;

	while (unchanged_borders < num_borders && cnt < 5) {
		cnt++;
		unchanged_borders = 0;
		for (int j = 0; j < num_borders; j++) {
			curr_border = borders[j];
			find_simple_point_candidates(img, curr_border, simple_border_points, LUT);
			no_change = true;
			num_border_points = simple_border_points.size();
			for (int i = 0; i < num_border_points; i++) {
				cv::Point3d target;
				target = simple_border_points[i];
				p = target.x;
				r = target.y;
				c = target.z;
				get_neighborhood(img, p, r, c, neighb);
				if (is_simple_point(neighb)) {
					img.at<cv::Vec3b>(p, r)[c] = 0;
					no_change = false;
				}
			if (no_change)
				unchanged_borders += 1;
			}
		}
	}
	//return img;
}

void find_simple_point_candidates(cv::Mat& img, int curr_border, vector<cv::Point3d>& simple_border_points, vector<int>& LUT) {
	//����
	int neighborhood[27];
	bool is_border_pt;

	int rows = img.rows;
	int cols = img.cols;
	int channels = img.channels();

	simple_border_points.clear();
	for (int p = 1; p < rows; p++) {
		for (int r = 1; r < cols; r++) {
			for (int c = 1; c < channels; c++) {
				//�ȼ��� 0�̸� ���ʿ� ����

				if (img.at<cv::Vec3b>(p, r)[c] != 1) continue;

				//������ �´��� Ȯ��
				is_border_pt = (curr_border == 1 && img.at<cv::Vec3b>(p, r)[c - 1] == 0 ||
					curr_border == 2 && img.at<cv::Vec3b>(p, r)[c + 1] == 0 ||
					curr_border == 3 && img.at<cv::Vec3b>(p, r + 1)[c] == 0 ||
					curr_border == 4 && img.at<cv::Vec3b>(p, r - 1)[c] == 0 ||
					curr_border == 5 && img.at<cv::Vec3b>(p + 1, r)[c] == 0 ||
					curr_border == 6 && img.at<cv::Vec3b>(p - 1, r)[c] == 0);
				if (!is_border_pt) continue;

				get_neighborhood(img, p, r, c, neighborhood);

				if (is_endpoint(neighborhood) || !is_Euler_invariant(neighborhood, LUT) || !is_simple_point(neighborhood))
					continue;

				cv::Point3d point(p, r, c);
				simple_border_points.push_back(point);
			}
		}
	}
}

void get_neighborhood(cv::Mat& img, int p, int r, int c, int* neighborhood) {
	//gpt api�� ��Ģ ��ȯ�� �κ� ����
	neighborhood[0] = img.at<cv::Vec3b>(p - 1, r - 1)[c - 1];
	neighborhood[1] = img.at<cv::Vec3b>(p - 1, r)[c - 1];
	neighborhood[2] = img.at<cv::Vec3b>(p - 1, r + 1)[c - 1];

	neighborhood[3] = img.at<cv::Vec3b>(p - 1, r - 1)[c];
	neighborhood[4] = img.at<cv::Vec3b>(p - 1, r)[c];
	neighborhood[5] = img.at<cv::Vec3b>(p - 1, r + 1)[c];

	neighborhood[6] = img.at<cv::Vec3b>(p - 1, r - 1)[c + 1];
	neighborhood[7] = img.at<cv::Vec3b>(p - 1, r)[c + 1];
	neighborhood[8] = img.at<cv::Vec3b>(p - 1, r + 1)[c + 1];

	neighborhood[9] = img.at<cv::Vec3b>(p, r - 1)[c - 1];
	neighborhood[10] = img.at<cv::Vec3b>(p, r)[c - 1];
	neighborhood[11] = img.at<cv::Vec3b>(p, r + 1)[c - 1];

	neighborhood[12] = img.at<cv::Vec3b>(p, r - 1)[c];
	neighborhood[13] = img.at<cv::Vec3b>(p, r)[c];
	neighborhood[14] = img.at<cv::Vec3b>(p, r + 1)[c];

	neighborhood[15] = img.at<cv::Vec3b>(p, r - 1)[c + 1];
	neighborhood[16] = img.at<cv::Vec3b>(p, r)[c + 1];
	neighborhood[17] = img.at<cv::Vec3b>(p, r + 1)[c + 1];

	neighborhood[18] = img.at<cv::Vec3b>(p + 1, r - 1)[c - 1];
	neighborhood[19] = img.at<cv::Vec3b>(p + 1, r)[c - 1];
	neighborhood[20] = img.at<cv::Vec3b>(p + 1, r + 1)[c - 1];

	neighborhood[21] = img.at<cv::Vec3b>(p + 1, r - 1)[c];
	neighborhood[22] = img.at<cv::Vec3b>(p + 1, r)[c];
	neighborhood[23] = img.at<cv::Vec3b>(p + 1, r + 1)[c];

	neighborhood[24] = img.at<cv::Vec3b>(p + 1, r - 1)[c + 1];
	neighborhood[25] = img.at<cv::Vec3b>(p + 1, r)[c + 1];
	neighborhood[26] = img.at<cv::Vec3b>(p + 1, r + 1)[c + 1];
}

bool is_endpoint(int* neighborhood) {
	int s = 0;
	for (int j = 0; j < 27; j++)
		s += neighborhood[j];

	return s == 2;
}

int _neighb_idx[8][7] = {
	{2, 1, 11, 10, 5, 4, 14},   // NEB
	{0, 9, 3, 12, 1, 10, 4},    // NWB
	{8, 7, 17, 16, 5, 4, 14},   // SEB
	{6, 15, 7, 16, 3, 12, 4},   // SWB
	{20, 23, 19, 22, 11, 14,  10}, // NEU
	{18, 21, 9, 12, 19, 22, 10}, // NWU
	{26, 23, 17, 14, 25, 22, 16}, // SEU
	{24, 25, 15, 16, 21, 22, 12}, // SWU
};

bool is_Euler_invariant(int* neighborhood, vector<int>& LUT) {
	int n, euler_char = 0;
	for (int o = 0; o < 8; o++) {
		n = 1;
		for (int j = 0; j < 7; j++) {
			int idx = _neighb_idx[o][j];
			if (neighborhood[idx] == 1)
				n |= 1 << (7 - j); //�̺κ��� ���Ǳ�� ������
		}
		euler_char += LUT[n];
	}
	return euler_char == 0;
}


bool is_simple_point(int* neighborhood) {
	int cube[26];
	for (int i = 0; i < 26; i++) {
		if (i < 13) cube[i] = neighborhood[i];
		else  cube[i] = neighborhood[i+1];
	}

	int label = 2;
	for (int i = 0; i < 26; i++) {
		if (cube[i] == 1) {
			if (i == 0 || i == 1 || i == 3 || i == 4 || i == 9 || i == 10 || i == 12)
				octree_labeling(1, label, cube);
			else if (i == 2 || i == 5 || i == 11 || i == 13)
				octree_labeling(2, label, cube);
			else if (i == 6 || i == 7 || i == 14 || i == 15)
				octree_labeling(3, label, cube);
			else if (i == 8 || i == 16)
				octree_labeling(4, label, cube);
			else if (i == 17 || i == 18 || i == 20 || i == 21)
				octree_labeling(5, label, cube);
			else if (i == 19 || i == 22)
				octree_labeling(6, label, cube);
			else if (i == 23 || i == 24)
				octree_labeling(7, label, cube);
			else if (i == 25)
				octree_labeling(8, label, cube);
			label += 1;
			if ((label - 2) >= 2)
				return false;
		}
		
	}
	return true;
}


//--------------------------------LOOK_UP_TABLE-----------------------------------
void fill_Euler_LUT(vector<int>& LUT) {
	int arr[128] = { 1, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, -1,
				 3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, 3, -1, 1, 1,
				 3, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1,
				 -1, 1, 1, -1, 3, 1, 1, -1, -7, -1, -1, 1, -3, -1, -1, 1, -1,
				 1, 1, -1, 3, 1, 1, -1, -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1,
				 -1, 3, 1, 1, -1, -3, 3, -1, 1, 1, 3, -1, 1, -1, 1, 1, -1, 3,
				 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1 };

	for (int i = 0; i < 128; i++) {
		LUT.push_back(0);
		LUT.push_back(arr[i]);
	}
}

//vec (pare(vec,vec(vec)))
std::vector<std::pair<std::vector<int>, std::vector<std::vector<int>>>> OC_TREE = {
	{{0, 1, 3, 4, 9, 10, 12},      {{}, {2}, {3}, {2, 3, 4}, {5}, {2, 5, 6}, {3, 5, 7}}}, // octant 1
	{{1, 4, 10, 2, 5, 11, 13},     {{1}, {1, 3, 4}, {1, 5, 6}, {}, {4}, {6}, {4, 6, 8}}}, // octant 2
	{{3, 4, 12, 6, 7, 14, 15},     {{1}, {1, 2, 4}, {1, 5, 7}, {}, {4}, {7}, {4, 7, 8}}}, // octant 3
	{{4, 5, 13, 7, 15, 8, 16},     {{1, 2, 3}, {2}, {2, 6, 8}, {3}, {3, 7, 8}, {}, {8}}}, // octant 4
	{{9, 10, 12, 17, 18, 20, 21},  {{1}, {1, 2, 6}, {1, 3, 7}, {}, {6}, {7}, {6, 7, 8}}}, // octant 5
	{{10, 11, 13, 18, 21, 19, 22}, {{1, 2, 5}, {2}, {2, 4, 8}, {5}, {5, 7, 8}, {}, {8}}}, // octant 6
	{{12, 14, 15, 20, 21, 23, 24}, {{1, 3, 5}, {3}, {3, 4, 8}, {5}, {5, 6, 8}, {}, {8}}}, // octant 7
	{{13, 15, 16, 21, 22, 24, 25}, {{2, 4, 6}, {3, 4, 7}, {4}, {5, 6, 7}, {6}, {7}, {}}}  // octant 8
};

void octree_labeling(int octant, int label, int *cube){
	for (int o = 1; o < 9; o++) {
		if (octant == o) {
			vector<int> indices = OC_TREE[o - 1].first;
			std::vector<std::vector<int>> list_octants = OC_TREE[o - 1].second;
			for (int i = 0; i < 7; i++) {
				int idx = indices[i];
				std::vector<int> new_octants = list_octants[i];
				if (cube[idx] == 1) {
					cube[idx] = label;
					for (int j = 0; j < new_octants.size(); j++) {
						int new_octant = new_octants[j];
						octree_labeling(new_octant, label, cube);
					}
				}
			}
		}
	}
}