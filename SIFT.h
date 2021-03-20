#pragma once
// #include "globals.h"
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

using namespace std;

class Keypoint
{
public:
	double			xi;
	double			yi;	// location of keypoints
	vector<double>	mag;	// The list of magnitudes at this point
	vector<double>	orien;	// The list of orientations detected
	int	scale;	// The scale where this was detected
	//constructors
	Keypoint() { }
	Keypoint(double x, double y) { xi = x; yi = y; }
	Keypoint(double x, double y, vector<double> const& m, vector<double> const& o, int s)
	{
		xi = x;
		yi = y;
		mag = m;
		orien = o;
		scale = s;
	}
};


class Descriptor:public Keypoint
{
public:
	vector<double>	fv;		// The feature vector (128 features)
	//constructors
	Descriptor()
	{
	}

	Descriptor(double x, double y, vector<double> const& f)
	{
		xi = x;
		yi = y;
		fv = f;
	}
};



class SIFT
{
/*
PRIVATE FUNCTION TO SUPPORT IN FINDING MATCHING POINTS BETWEEN 2 IMAGES
*/
private:
	//calculate distance of two vector
	double euclidDistance(vector<double> vec1, vector<double> vec2);
	// kNN algorithm
	// vector<Descriptor> descriptors has many vector
	// so this function will find the vector of descriptors has the minimum distance with v
	//and return the index of this vector in descriptors
	int kNearestNeighbor(vector<double> v, vector<Descriptor> descriptors);
	//find pairs matched
	void findPairs(vector<Descriptor> descriptors1, vector<Descriptor> descriptors2,
		vector<pair<Keypoint, Keypoint> >& match);
public:
	//matching two images
	void matchingImage(Mat img1, Mat img2);
/*
PRIVATE FUNCTION TO FIND INTERESTING POINTS
*/
private:
	//create scale space
	void BuildScaleSpace();
	// detect maxima and minima
	void DetectExtrema(); 
	//calculate orientation of a keypoint
	void AssignOrientations();
	//find feature vector of a keypoint
	void ComputeKeypointDescriptors();
	//
	//find kernel size with a specific sigma value
    int GetKernelSize(double sigma, double epsilon = 0.001);
	// build gaussian table with sigma and size of kernel
	Mat BuildGaussianTable(int size, double sigma);
public:
	//Show interesting points on images
	Mat ShowKeypoints();
	//SIFT algorithm
	void DoSift();
public:
	SIFT(Mat img, int octaves, int scales);
	SIFT() {}
	~SIFT();

	//get keypoints
	vector<Keypoint> getKeyPoints()
	{
		return m_keyPoints;
	}
	//get discriptors
	vector<Descriptor> getDescriptors()
	{
		return m_keyDescs;
	}
	/*VARIABLES*/
private:
	Mat m_srcImage;		// The image we're working on
	int m_numOctaves;	// The number of octaves
	int m_numScales;	// The number of scales
	int m_numKeypoints;	// The number of keypoints

	Mat** m_gImages;		// stores the gaussian blurred images
	Mat** m_dogImages;	// stores the different of Gaussian DoG images
	Mat** m_extremaImages;	// stores binary images which used to find extrema
						// In the binary image, 1 = extrema, 0 = not extrema
	double** m_Sigmas;	// stores the sigma used to blur a particular image

	vector<Keypoint> m_keyPoints;	// stores keypoints
	vector<Descriptor> m_keyDescs;	// stores keypoint's descriptor
};

