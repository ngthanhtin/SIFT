#include "SIFT.h"

#define CURVATURE_THRESHOLD		10.0
#define CONTRAST_THRESHOLD		0.03
#define NUM_BINS				36
#define FEATURE_WINDOW_SIZE		16
#define DESC_NUM_BINS			8
#define FVSIZE					128
#define	FV_THRESHOLD			0.2


SIFT::SIFT(Mat img, int octaves, int scales)
{
	m_srcImage = img.clone();
	// Set the number of octaves and intervals
	m_numOctaves = octaves;
	m_numScales = scales;
	// Create an array of gaussian blurred images
	m_gImages = new Mat*[m_numOctaves];
	for (int i = 0; i < m_numOctaves; i++)
		m_gImages[i] = new Mat[m_numScales + 3];

	// Create an array to store images generated after the DoG operation
	m_dogImages = new Mat*[m_numOctaves];
	for (int i = 0; i < m_numOctaves; i++)
		m_dogImages[i] = new Mat[m_numScales + 2];

	// Create an array that will hold if a particular point is an extrema or not
	m_extremaImages = new Mat*[m_numOctaves];
	for (int i = 0; i < m_numOctaves; i++)
		m_extremaImages[i] = new Mat[m_numScales];

	// Create an array to hold the sigma value used to blur the gaussian images.
	m_Sigmas = new double*[m_numOctaves];
	for (int i = 0; i<m_numOctaves; i++)
		m_Sigmas[i] = new double[m_numScales + 3];
}


SIFT::~SIFT()
{
	for (int i = 0; i < m_numOctaves; i++)
	{
		// Delete memory for that array
		delete[] m_gImages[i];
		delete[] m_dogImages[i];
		delete[] m_extremaImages[i];
		delete[] m_Sigmas[i];
	}
	//Delete the 2D arrays
	delete[] m_gImages;
	delete[] m_dogImages;
	delete[] m_extremaImages;
	delete[] m_Sigmas;
}
void SIFT::matchingImage(Mat img1, Mat img2)
{
	//create sift opretator for 2 image with the same octaves and scales
	int octaves = 3;
	int scale = 4;
	SIFT sift1(img1, octaves, scale), sift2(img2, octaves, scale);
	//do sift algorithm
	sift1.DoSift();
	sift2.DoSift();
	//vector stored point pairs matched
	vector<pair<Keypoint, Keypoint> > match;
	findPairs(sift1.getDescriptors(), sift2.getDescriptors(), match);
	//show keypoints on each image and draw the lines between two points matched.
	// images with keypoints
	Mat keypoints1, keypoints2;
	keypoints1 = sift1.ShowKeypoints();
	keypoints2 = sift2.ShowKeypoints();
	// appended image
	Mat appended; // we will join two image into a big image
	int max_rows; // we join 2 image in horizontal, so we must find the max height (means max rows)
	if (keypoints1.rows > keypoints2.rows)
	{
		max_rows = keypoints1.rows;
	}
	else
	{
		max_rows = keypoints2.rows;
	}
	// join 2 images which is showed keypoints
	appended.create(max_rows, keypoints1.cols + keypoints2.cols, keypoints1.type());
	keypoints1.copyTo(appended(Range(0, keypoints1.rows), Range(0, keypoints1.cols)));
	keypoints2.copyTo(appended(Range(0, keypoints2.rows), Range(keypoints1.cols, keypoints1.cols + keypoints2.cols)));
	//draw line between point-pairs matched
	for (int i = 0; i < match.size(); i++)
	{
		Point p1 = Point(match[i].first.xi, match[i].first.yi);
		Point p2 = Point(match[i].second.xi + keypoints1.cols, match[i].second.yi);
		line(appended, p1, p2, Scalar(0, 255, 0), 1);
	}
	imshow("matched", appended);
}

double SIFT::euclidDistance(vector<double> vec1, vector<double> vec2)
{
	//calculate euclid distance of two vectors
	double sum = 0.0f;
	int cols = vec1.size();
	for (int i = 0; i < cols; i++)
	{
		// (x1-x2)^2
		sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	}
	return sqrt(sum);
}

// Find the index of nearest neighbor point from keypoints.
int SIFT::kNearestNeighbor(vector<double> v, vector<Descriptor> descriptors)
{
	int neighbor = -1;
	double minDist = 1e6;

	for (int i = 0; i < descriptors.size(); i++)
	{
		vector<double> v2 = descriptors[i].fv;
		double d = euclidDistance(v, v2);
		if (d < minDist)
		{
			minDist = d;
			neighbor = i;
		}
	}

	if (minDist < 400)
	{
		return neighbor;
	}

	return -1;
}

void SIFT::findPairs(vector<Descriptor> descriptors1,
	vector<Descriptor> descriptors2,
	vector<pair<Keypoint, Keypoint> >& match)
{
	int sizeDes1 = descriptors1.size(); // size of descriptor 1
	for (int i = 0; i < sizeDes1; i++)
	{
		vector<double> desc1 = descriptors1[i].fv;
		int nn = kNearestNeighbor(desc1, descriptors2);
		if (nn >= 0)
		{
			Keypoint dest(descriptors2[nn].xi, descriptors2[nn].yi);
			Keypoint src(descriptors1[i].xi, descriptors1[i].yi);
			match.push_back(make_pair(src, dest));
		}
	}
}

void SIFT::DoSift()
{
	BuildScaleSpace();
	DetectExtrema();
	AssignOrientations();
	ComputeKeypointDescriptors();
}



void SIFT::BuildScaleSpace()
{
	Mat greyscaleImg = convertRgbToGrayscale(m_srcImage);
	
	// normalize the pixel value from 0..255 into 0..1
	for (int r = 0; r < greyscaleImg.rows; r++)
	{
		for (int c = 0; c < greyscaleImg.cols; c++)
		{
			greyscaleImg.at<double>(r, c) = greyscaleImg.at<double>(r, c) / 255.f;
		}
	}
	// Lowe claims blur the image with a sigma of 0.5 and double it's dimensions
	// to increase the number of stable keypoints
	GaussianBlur(greyscaleImg, greyscaleImg, Size(0, 0), 0.5);

	// Create an image double the dimensions, resize imgGray and store it in m_gImage[0][0]
	m_gImages[0][0] = Mat(Size(greyscaleImg.rows * 2, greyscaleImg.cols * 2), CV_64F);
	resize(greyscaleImg, m_gImages[0][0], m_gImages[0][0].size());

	double initSigma = sqrt(2.0f);

	// Keep a track of the sigmas
	m_Sigmas[0][0] = initSigma * 0.5;

	// Now for the actual image generation
	for (int i = 0; i<m_numOctaves; i++)
	{
		// Reset sigma for each octave
		double sigma = initSigma;
		Size currentSize = Size(m_gImages[i][0].cols, m_gImages[i][0].rows);

		for (int j = 1; j < m_numScales + 3; j++)
		{
			m_gImages[i][j] = Mat(currentSize, CV_64F);

			// Calculate a sigma to blur the current image to get the next one
			double sigma_step = sqrt(pow(2.0, 2.0 / m_numScales) - 1) * sigma;
			sigma = pow(2.0, 1.0 / m_numScales) * sigma;

			// Store sigma values
			m_Sigmas[i][j] = sigma * 0.5 * pow(2.0f, (double)i);
			// Apply gaussian smoothing
			GaussianBlur(m_gImages[i][j - 1], m_gImages[i][j], Size(0, 0), sigma_step);
			// Calculate the DoG image
			m_dogImages[i][j - 1] = m_gImages[i][j - 1] - m_gImages[i][j];
		}

		// If we're not at the last octave
		if (i<m_numOctaves - 1)
		{
			// Reduce size to half
			currentSize.width /= 2;
			currentSize.height /= 2;

			// resize the image
			m_gImages[i + 1][0] = Mat(currentSize, CV_64F);
			resize(m_gImages[i][0], m_gImages[i + 1][0], m_gImages[i + 1][0].size());
			m_Sigmas[i + 1][0] = m_Sigmas[i][m_numScales];

		}
	}
}
// DetectExtrema()
void SIFT::DetectExtrema()
{
	//uses to calculate principal curvatures
	double curvature_ratio, curvature_threshold;	
	curvature_threshold
		= (CURVATURE_THRESHOLD + 1)*(CURVATURE_THRESHOLD + 1) / CURVATURE_THRESHOLD;
	//uses to store value of hessian matrix
	double dxx, dyy, dxy, trH, detH;
	//
	Mat middle, up, down;
	int num = 0; // the variable to store the number of keypoins detected

	for (int i = 0; i < m_numOctaves; i++)
	{
		for (int j = 1; j < m_numScales + 1; j++)
		{
			// set all points of matrix extrema to zero
			m_extremaImages[i][j - 1] = Mat(m_dogImages[i][0].size(), CV_64F, Scalar::all(0));

			middle = m_dogImages[i][j].clone(); // each pixel in this image will be compared
											  // with 8 adjacent in this image and 9 points in image above,
			                                  // and 9 point in image below
			up = m_dogImages[i][j + 1].clone(); // image above
			down = m_dogImages[i][j - 1].clone(); // image below

			for (int xi = 1; xi < m_dogImages[i][j].size().width - 1; xi++)
			{
				for (int yi = 1; yi < m_dogImages[i][j].size().height - 1; yi++)
				{
					// true if a keypoint is a maxima/minima
					// but needs to be tested for contrast/edge thingy
					bool isExtrema = false;

					double currentPixel = middle.at<double>(yi, xi);
					
					// Check for a maximum
					if (currentPixel > middle.at<double>(yi - 1, xi) &&
						currentPixel > middle.at<double>(yi + 1, xi) &&
						currentPixel > middle.at<double>(yi, xi - 1) &&
						currentPixel > middle.at<double>(yi, xi + 1) &&
						currentPixel > middle.at<double>(yi - 1, xi - 1) &&
						currentPixel > middle.at<double>(yi - 1, xi + 1) &&
						currentPixel > middle.at<double>(yi + 1, xi + 1) &&
						currentPixel > middle.at<double>(yi + 1, xi - 1) &&
						currentPixel > up.at<double>(yi, xi) &&
						currentPixel > up.at<double>(yi - 1, xi) &&
						currentPixel > up.at<double>(yi + 1, xi) &&
						currentPixel > up.at<double>(yi, xi - 1) &&
						currentPixel > up.at<double>(yi, xi + 1) &&
						currentPixel > up.at<double>(yi - 1, xi - 1) &&
						currentPixel > up.at<double>(yi - 1, xi + 1) &&
						currentPixel > up.at<double>(yi + 1, xi + 1) &&
						currentPixel > up.at<double>(yi + 1, xi - 1) &&
						currentPixel > down.at<double>(yi, xi) &&
						currentPixel > down.at<double>(yi - 1, xi) &&
						currentPixel > down.at<double>(yi + 1, xi) &&
						currentPixel > down.at<double>(yi, xi - 1) &&
						currentPixel > down.at<double>(yi, xi + 1) &&
						currentPixel > down.at<double>(yi - 1, xi - 1) &&
						currentPixel > down.at<double>(yi - 1, xi + 1) &&
						currentPixel > down.at<double>(yi + 1, xi + 1) &&
						currentPixel > down.at<double>(yi + 1, xi - 1))
					{
						m_extremaImages[i][j - 1].at<double>(yi, xi) = 255.f;
						num++;
						isExtrema = true;
					}
					// Check if it's a minimum
					else if (currentPixel < middle.at<double>(yi - 1, xi) &&
						currentPixel < middle.at<double>(yi + 1, xi) &&
						currentPixel < middle.at<double>(yi, xi - 1) &&
						currentPixel < middle.at<double>(yi, xi + 1) &&
						currentPixel < middle.at<double>(yi - 1, xi - 1) &&
						currentPixel < middle.at<double>(yi - 1, xi + 1) &&
						currentPixel < middle.at<double>(yi + 1, xi + 1) &&
						currentPixel < middle.at<double>(yi + 1, xi - 1) &&
						currentPixel < up.at<double>(yi, xi) &&
						currentPixel < up.at<double>(yi - 1, xi) &&
						currentPixel < up.at<double>(yi + 1, xi) &&
						currentPixel < up.at<double>(yi, xi - 1) &&
						currentPixel < up.at<double>(yi, xi + 1) &&
						currentPixel < up.at<double>(yi - 1, xi - 1) &&
						currentPixel < up.at<double>(yi - 1, xi + 1) &&
						currentPixel < up.at<double>(yi + 1, xi + 1) &&
						currentPixel < up.at<double>(yi + 1, xi - 1) &&
						currentPixel < down.at<double>(yi, xi) &&
						currentPixel < down.at<double>(yi - 1, xi) &&
						currentPixel < down.at<double>(yi + 1, xi) &&
						currentPixel < down.at<double>(yi, xi - 1) &&
						currentPixel < down.at<double>(yi, xi + 1) &&
						currentPixel < down.at<double>(yi - 1, xi - 1) &&
						currentPixel < down.at<double>(yi - 1, xi + 1) &&
						currentPixel < down.at<double>(yi + 1, xi + 1) &&
						currentPixel < down.at<double>(yi + 1, xi - 1))
					{
						m_extremaImages[i][j - 1].at<double>(yi, xi) = 255.f;
						num++;
						isExtrema = true;
					}

					// The contrast check
					if (isExtrema && fabs(middle.at<double>(yi, xi)) < CONTRAST_THRESHOLD)
					{
						m_extremaImages[i][j - 1].at<double>(yi, xi) = 0.f;
						num--;

						isExtrema = false;
					}

					// The edge check
					if (isExtrema)
					{
						dxx = (middle.at<double>(yi - 1, xi) +
							middle.at<double>(yi + 1, xi) -
							2.0*middle.at<double>(yi, xi));

						dyy = (middle.at<double>(yi, xi - 1) +
							middle.at<double>(yi, xi + 1) -
							2.0*middle.at<double>(yi, xi));

						dxy = (middle.at<double>(yi - 1, xi - 1) +
							middle.at<double>(yi + 1, xi + 1) -
							middle.at<double>(yi + 1, xi - 1) -
							middle.at<double>(yi - 1, xi + 1)) / 4.0;

						trH = dxx + dyy;
						detH = dxx*dyy - dxy*dxy;

						curvature_ratio = trH*trH / detH;
						if (detH<0 || curvature_ratio > curvature_threshold)
						{
							m_extremaImages[i][j - 1].at<double>(yi, xi) = 0.f;
							num--;

							isExtrema = false;
						}
					}
				}
			}

		}
	}
	m_numKeypoints = num; // the number of keypoints detected
}

void SIFT::AssignOrientations()
{
	int i, j, k, xi, yi;
	int kk, tt;

	// These images hold the magnitude and direction of gradient 
	// for all blurred images
	Mat** magnitude = new Mat*[m_numOctaves];
	Mat** orientation = new Mat*[m_numOctaves];

	for (i = 0; i < m_numOctaves; i++)
	{
		magnitude[i] = new Mat[m_numScales];
		orientation[i] = new Mat[m_numScales];
	}

	// These two loops are to calculate the magnitude and orientation of gradients
	// through all octaces once and for all. We don't run around calculating things
	// again and again that way.

	// Iterate through all octaves
	for (i = 0; i<m_numOctaves; i++)
	{
		// Iterate through all scales
		for (j = 1; j<m_numScales + 1; j++)
		{
			magnitude[i][j - 1] = Mat(m_gImages[i][j].size(), CV_64F, Scalar::all(0));
			orientation[i][j - 1] = Mat(m_gImages[i][j].size(), CV_64F, Scalar::all(0));

			// Iterate over the gaussian image with the current octave and interval
			for (xi = 1; xi < m_gImages[i][j].cols - 1; xi++)
			{
				for (yi = 1; yi < m_gImages[i][j].rows - 1; yi++)
				{
					// Calculate gradient
					double dx = 
						m_gImages[i][j].at<double>(yi, xi + 1) - m_gImages[i][j].at<double>(yi, xi - 1);
					double dy =
						m_gImages[i][j].at<double>(yi + 1, xi) - m_gImages[i][j].at<double>(yi - 1, xi);

					// Store magnitude
					magnitude[i][j - 1].at<double>(yi, xi) = sqrt(dx*dx + dy*dy);

					// Store orientation as radians
					double ori = atan(dy / dx);
					orientation[i][j - 1].at<double>(yi, xi) = ori;
				}
			}
		}
	}

	// The histogram with 8 bins
	double* hist_orient = new double[NUM_BINS];

	// Go through all octaves
	for (i = 0; i < m_numOctaves; i++)
	{
		// Store current scale, width and height
		int scale = (int)pow(2.0, (double)i);
		int width = m_gImages[i][0].cols;
		int height = m_gImages[i][0].rows;

		// Go through all intervals in the current scale
		for (j = 1; j < m_numScales + 1; j++)
		{
			double sigma = m_Sigmas[i][j];

			// This is used for magnitudes
			Mat imgWeight = Mat(Size(width, height), CV_64F, Scalar::all(1));
			GaussianBlur(magnitude[i][j - 1], imgWeight, Size(0, 0), 1.5*sigma);

			// Get the kernel size for the Guassian blur
			int half_size = GetKernelSize(1.5*sigma) / 2;

			// Temporarily used to generate a mask of region used to calculate 
			// the orientations
			Mat imgMask = Mat(Size(width, height), CV_8UC1, Scalar::all(0));

			// Iterate through all points at this octave and scales
			for (xi = 0; xi < width; xi++)
			{
				for (yi = 0; yi < height; yi++)
				{
					//if it's a keypoint
					if (m_extremaImages[i][j - 1].at<double>(yi, xi) != 0)
					{
						// Reset the histogram thingy
						for (k = 0; k < NUM_BINS; k++)
							hist_orient[k] = 0.0;

						// Go through all pixels in the window around the extrema
						for (kk = -half_size; kk <= half_size; kk++)
						{
							for (tt = -half_size; tt <= half_size; tt++)
							{
								// Ensure we're within the image
								if (xi + kk<0 || xi + kk >= width || yi + tt<0 || yi + tt >= height)
									continue;

								double sampleOrient = orientation[i][j - 1].at<double>(yi + tt, xi + kk);

								if (sampleOrient <= -M_PI || sampleOrient > M_PI)
									cout << "Bad Orientation: " << sampleOrient << "\n";

								sampleOrient += M_PI;

								// Convert to degrees
								int sampleOrientDegrees = sampleOrient * 180 / M_PI;
								int bin = (int)sampleOrientDegrees / (360 / NUM_BINS);
								hist_orient[bin] += imgWeight.at<double>(yi + tt, xi + kk);
								imgMask.at<uchar>(yi + tt, xi + kk) = 255;
							}
						}

						// We've computed the histogram. Now check for the maximum
						double max_peak = hist_orient[0];
						int max_peak_index = 0;
						for (k = 1; k < NUM_BINS; k++)
						{
							if (hist_orient[k] > max_peak)
							{
								max_peak = hist_orient[k];
								max_peak_index = k;
							}
						}

						// List of magnitudes and orientations at the current extrema
						vector<double> orien;
						vector<double> mag;
						for (k = 0; k < NUM_BINS; k++)
						{
							// Do we have a good peak?
							if (hist_orient[k]> 0.8*max_peak)
							{
								// Three points. (x2,y2) is the peak and (x1,y1)
								// and (x3,y3) are the neigbours to the left and right.
								// If the peak occurs at the extreme left, the "left
								// neighbour" is equal to the right most. Similarly for
								// the other case (peak is rightmost)
								double x1 = k - 1;
								double y1;
								double x2 = k;
								double y2 = hist_orient[k];
								double x3 = k + 1;
								double y3;

								if (k == 0)
								{
									y1 = hist_orient[NUM_BINS - 1];
									y3 = hist_orient[1];
								}
								else if (k == NUM_BINS - 1)
								{
									y1 = hist_orient[NUM_BINS - 1];
									y3 = hist_orient[0];
								}
								else
								{
									y1 = hist_orient[k - 1];
									y3 = hist_orient[k + 1];
								}

								
								// find 3 points using larange operator
								// A downward parabola has the general form
								//
								// y = a * x^2 + bx + c
								// Now the three equations stem from the three points
								// (x1,y1) (x2,y2) (x3.y3) are
								//
								// y1 = a * x1^2 + b * x1 + c
								// y2 = a * x2^2 + b * x2 + c
								// y3 = a * x3^2 + b * x3 + c
								//
								// in Matrix notation, this is y = Xb, where
								// y = (y1 y2 y3)' b = (a b c)' and
								// 
								//     x1^2 x1 1
								// X = x2^2 x2 1
								//     x3^2 x3 1
								//
								// OK, we need to solve this equation for b
								// this is done by inverse the matrix X
								//
								// b = inv(X) Y

								double *b = new double[3];
								Mat X = Mat(3, 3, CV_64F);
								Mat matInv = Mat(3, 3, CV_64F);

								X.at<double>(0, 0) = x1*x1;
								X.at<double>(1, 0) = x1;
								X.at<double>(2, 0) = 1;

								X.at<double>(0, 1) = x2*x2;
								X.at<double>(1, 1) = x2;
								X.at<double>(2, 1) = 1;

								X.at<double>(0, 2) = x3*x3;
								X.at<double>(1, 2) = x3;
								X.at<double>(2, 2) = 1;

								// Invert the matrix
								invert(X, matInv);

								b[0] = 
									matInv.at<double>(0, 0)*y1 + matInv.at<double>(1, 0)*y2 + matInv.at<double>(2, 0)*y3;
								b[1] =
									matInv.at<double>(0, 1)*y1 + matInv.at<double>(1, 1)*y2 + matInv.at<double>(2, 1)*y3;
								b[2] =
									matInv.at<double>(0, 2)*y1 + matInv.at<double>(1, 2)*y2 + matInv.at<double>(2, 2)*y3;

								double x0 = -b[1] / (2 * b[0]);

								// Anomalous situation
								if (fabs(x0)>2 * NUM_BINS)
									x0 = x2;

								while (x0<0)
									x0 += NUM_BINS;
								while (x0 >= NUM_BINS)
									x0 -= NUM_BINS;

								// Normalize it
								double x0_n = x0*(2 * M_PI / NUM_BINS);

								x0_n -= M_PI;

								orien.push_back(x0_n);
								mag.push_back(hist_orient[k]);
							}
						}

						// Save this keypoint into the list
						m_keyPoints.push_back(Keypoint(xi*scale / 2, yi*scale / 2, mag, orien, i*m_numScales + j - 1));
					}
				}
			}
		}
	}

	// Finally, deallocate memory
	for (i = 0; i<m_numOctaves; i++)
	{
		delete[] magnitude[i];
		delete[] orientation[i];
	}

	delete[] magnitude;
	delete[] orientation;
}

void SIFT::ComputeKeypointDescriptors()
{
	// magnitudes and orientations
	Mat** imgInterpolatedMagnitude = new Mat*[m_numOctaves];
	Mat** imgInterpolatedOrientation = new Mat*[m_numOctaves];
	for (int i = 0; i < m_numOctaves; i++)
	{
		imgInterpolatedMagnitude[i] = new Mat[m_numScales];
		imgInterpolatedOrientation[i] = new Mat[m_numScales];
	}

	// These two loops calculate the interpolated thingy for all octaves
	// and subimages
	for (int i = 0; i < m_numOctaves; i++)
	{
		for (int j = 1; j < m_numScales + 1; j++)
		{
			int width = m_gImages[i][j].size().width;
			int height = m_gImages[i][j].size().height;

			// Create an image
			Mat imgTemp = Mat(Size(width * 2, height * 2), CV_64F, Scalar::all(0));

			// Scale it up. This will give us "access" to in betweens
			resize(m_gImages[i][j], imgTemp, imgTemp.size());

			imgInterpolatedMagnitude[i][j - 1] = Mat(Size(width + 1, height + 1), CV_64F, Scalar::all(0));
			imgInterpolatedOrientation[i][j - 1] = Mat(Size(width + 1, height + 1), CV_64F, Scalar::all(0));

			// Do the calculations
			for (double ii = 1.5; ii<width - 1.5; ii++)
			{
				for (double jj = 1.5; jj<height - 1.5; jj++)
				{
					// "inbetween" change
					double dx = (m_gImages[i][j].at<double>(jj, ii + 1.5) + m_gImages[i][j].at<double>(jj, ii + 0.5)) / 2 - (m_gImages[i][j].at<double>(jj, ii - 1.5) + m_gImages[i][j].at<double>(jj, ii - 0.5)) / 2;
					double dy = (m_gImages[i][j].at<double>(jj + 1.5, ii) + m_gImages[i][j].at<double>(jj + 0.5, ii)) / 2 - (m_gImages[i][j].at<double>(jj - 1.5, ii) + m_gImages[i][j].at<double>(jj - 0.5, ii)) / 2;

					int iii = ii + 1;
					int jjj = jj + 1;

					// Set the magnitude and orientation
					imgInterpolatedMagnitude[i][j - 1].at<double>(jjj, iii) =  sqrt(dx*dx + dy*dy);
					imgInterpolatedMagnitude[i][j - 1].at<double>(jjj, iii) =  (atan2(dy, dx) == M_PI) ? -M_PI : atan2(dy, dx);
				}
			}

			// Pad the edges with zeros
			for (int iii = 0; iii<width + 1; iii++)
			{
				imgInterpolatedMagnitude[i][j - 1].at<double>(0, iii) =  0;
				imgInterpolatedMagnitude[i][j - 1].at<double>(height, iii) = 0;
				imgInterpolatedMagnitude[i][j - 1].at<double>(0, iii) = 0;
				imgInterpolatedMagnitude[i][j - 1].at<double>(height, iii) = 0;
			}

			for (int jjj = 0; jjj<height + 1; jjj++)
			{
				imgInterpolatedMagnitude[i][j - 1].at<double>(jjj, 0) = 0;
				imgInterpolatedMagnitude[i][j - 1].at<double>(jjj, width) = 0;
				imgInterpolatedMagnitude[i][j - 1].at<double>(jjj, 0) = 0;
				imgInterpolatedMagnitude[i][j - 1].at<double>(jjj, width) = 0;
			}




		}
	}

	// Build an Interpolated Gaussian Table of size FEATURE_WINDOW_SIZE
	// Lowe suggests sigma should be half the window size
	// This is used to construct the "circular gaussian window" to weight 
	// magnitudes
	Mat G = BuildGaussianTable(FEATURE_WINDOW_SIZE, 0.5 * FEATURE_WINDOW_SIZE);

	vector<double> hist(DESC_NUM_BINS);

	// Loop over all keypoints
	for (int ikp = 0; ikp < m_numKeypoints; ikp++)
	{
		int scale = m_keyPoints[ikp].scale;
		double kpxi = m_keyPoints[ikp].xi;
		double kpyi = m_keyPoints[ikp].yi;

		double descxi = kpxi;
		double descyi = kpyi;

		int ii = (int)(kpxi * 2) / (int)(pow(2.0, (double)scale / m_numScales));
		int jj = (int)(kpyi * 2) / (int)(pow(2.0, (double)scale / m_numScales));

		int width = m_gImages[scale / m_numScales][0].size().width;
		int height = m_gImages[scale / m_numScales][0].size().height;

		vector<double> orien = m_keyPoints[ikp].orien;
		vector<double> mag = m_keyPoints[ikp].mag;
		

		// Find the orientation and magnitude that have the maximum impact
		// on the feature
		double main_mag = mag[0];
		double main_orien = orien[0];
		for (int orient_count = 1; orient_count<mag.size(); orient_count++)
		{
			if (mag[orient_count] > main_mag)
			{
				main_orien = orien[orient_count];
				main_mag = mag[orient_count];
			}
		}

		int half_size = FEATURE_WINDOW_SIZE / 2;
		Mat weight = Mat(FEATURE_WINDOW_SIZE, FEATURE_WINDOW_SIZE, CV_64F);
		vector<double> fv(FVSIZE);

		for (int i = 0; i < FEATURE_WINDOW_SIZE; i++)
		{
			for (int j = 0; j<FEATURE_WINDOW_SIZE; j++)
			{
				if (ii + i + 1 < half_size || ii + i + 1>width + half_size || jj + j + 1 < half_size || jj + j + 1 > height + half_size)
					weight.at<double>(j, i) = 0;
				else
					weight.at<double>(j, i) =
					G.at<double>(j, i)*imgInterpolatedMagnitude[scale / m_numScales][scale%m_numScales].at<double>(jj + j + 1 - half_size, ii + i + 1 - half_size);
			}
		}

		// Now that we've weighted the required magnitudes, we proceed to generating
		// the feature vector

		// The next two two loops are for splitting the 16x16 window
		// into sixteen 4x4 blocks
		for (int i = 0; i < FEATURE_WINDOW_SIZE / 4; i++)			// 4x4 thingy
		{
			for (int j = 0; j < FEATURE_WINDOW_SIZE / 4; j++)
			{
				// Clear the histograms
				for (int t = 0; t<DESC_NUM_BINS; t++)
					hist[t] = 0.0;

				// Calculate the coordinates of the 4x4 block
				int starti = (int)ii - (int)half_size + 1 + (int)(half_size / 2 * i);
				int startj = (int)jj - (int)half_size + 1 + (int)(half_size / 2 * j);
				int limiti = (int)ii + (int)(half_size / 2)*((int)(i)-1);
				int limitj = (int)jj + (int)(half_size / 2)*((int)(j)-1);

				// Go though this 4x4 block and do the thingy
				for (int k = starti; k <= limiti; k++)
				{
					for (int t = startj; t <= limitj; t++)
					{
						if (k<0 || k>width || t<0 || t>height)
							continue;

						// This is where rotation invariance is done
						double sample_orien =
							imgInterpolatedOrientation[scale / m_numScales][scale%m_numScales].at<double>(t, k);
						sample_orien -= main_orien;

						while (sample_orien<0)
							sample_orien += 2 * M_PI;

						while (sample_orien>2 * M_PI)
							sample_orien -= 2 * M_PI;

						
						if (!(sample_orien >= 0 && sample_orien < 2 * M_PI))
							cout << "BAD: " << sample_orien << "\n";

						int sample_orien_d = sample_orien * 180 / M_PI;

						int bin = sample_orien_d / (360 / DESC_NUM_BINS);// The bin
						double bin_f = (double)sample_orien_d / (double)(360 / DESC_NUM_BINS);// The actual entry


						// Add to the bin
						hist[bin] += (1 - fabs(bin_f - (bin + 0.5))) * weight.at<double>(t + half_size - 1 - jj, k + half_size - 1 - ii);
					}
				}

				// Keep adding these numbers to the feature vector
				for (int t = 0; t < DESC_NUM_BINS; t++)
				{
					fv[(i*FEATURE_WINDOW_SIZE / 4 + j)*DESC_NUM_BINS + t] = hist[t];
				}
			}
		}

		// Now, normalize the feature vector to ensure illumination independence
		double norm = 0;
		for (int t = 0; t < FVSIZE; t++)
			norm += pow(fv[t], 2.0);
		norm = sqrt(norm);

		for (int t = 0; t<FVSIZE; t++)
			fv[t] /= norm;

		// Now, threshold the vector
		for (int t = 0; t < FVSIZE; t++)
			if (fv[t]>FV_THRESHOLD)
				fv[t] = FV_THRESHOLD;

		// Normalize yet again
		norm = 0;
		for (int t = 0; t < FVSIZE; t++)
			norm += pow(fv[t], 2.0);
		norm = sqrt(norm);

		for (int t = 0; t < FVSIZE; t++)
			fv[t] /= norm;

		// We're done with this descriptor. Store it into a list
		m_keyDescs.push_back(Descriptor(descxi, descyi, fv));
	}


	//Deallocate memory
	for (int i = 0; i<m_numOctaves; i++)
	{
		delete[] imgInterpolatedMagnitude[i];
		delete[] imgInterpolatedOrientation[i];
	}

	delete[] imgInterpolatedMagnitude;
	delete[] imgInterpolatedOrientation;
}

int SIFT::GetKernelSize(double sigma, double epsilon)
{
	int times = 20; // max iteration is 20 times
	int i;
	for (i = 0; i < times; i++)
		if (exp(-((double)(i*i)) / (2.0*sigma*sigma)) < epsilon)
			break;
	int size = 2 * i - 1;
	return size;
}

Mat SIFT::BuildGaussianTable(int size, double sigma)
{
	int i, j;
	double half_kernel_size = size / 2 - 0.5;

	double sog = 0;
	Mat ret = Mat(size, size, CV_64F);


	double temp = 0;
	for (i = 0; i<size; i++)
	{
		for (j = 0; j<size; j++)
		{
			double x, y;
			x = i - half_kernel_size;
			y = j - half_kernel_size;
			temp = 1.0 / (2 * M_PI*sigma*sigma) * exp(-(x*x + y*y) / (2.0*sigma*sigma));
			ret.at<double>(j, i) =  temp;
			sog += temp;
		}
	}

	for (i = 0; i<size; i++)
	{
		for (j = 0; j<size; j++)
		{
			ret.at<double>(j, i) =  1.0 / sog * ret.at<double>(j, i);
		}
	}

	return ret;
}
Mat SIFT::ShowKeypoints()
{
	Mat img = m_srcImage.clone();

	for (int i = 0; i<m_numKeypoints; i++)
	{
		Keypoint kp = m_keyPoints[i];
		line(img, Point(kp.xi, kp.yi), Point(kp.xi, kp.yi), Scalar(255, 0, 0), 3);
		line(img, Point(kp.xi, kp.yi),
			Point(kp.xi + 10 * cos(kp.orien[0]), kp.yi + 10 * sin((double)kp.orien[0])),
			Scalar(255, 0, 0), 1);
	}
	return img;
	
}


