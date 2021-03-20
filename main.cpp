#include <highgui.h>
#include <iostream>

int main()
{
	// Create an instance of SIFT
	SIFT *sift = new SIFT("man1.jpg", 4, 2);

	sift->DoSift();				// Find keypoints
	sift->ShowKeypoints();		// Show the keypoints
	cvWaitKey(0);				// Wait for a keypress

	// Cleanup and exit
	delete sift;
	return 0;
}