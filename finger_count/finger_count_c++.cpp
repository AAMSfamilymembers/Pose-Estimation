#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <math.h>


using namespace cv;
using namespace std;


int main()
{
	VideoCapture cap(0); // open the default camera
	if(!cap.isOpened())  // check if we succeeded
        {
		cout << "Error in using VideoCapture function." << endl;	
		return -1;
	}
	
	String face_cascade_name = "/home/abhay/opencv/data/haarcascades/haarcascade_frontalface_default.xml";
	//String palm_cascade_name = "/home/abhay/palm.xml";
 	CascadeClassifier palm_cascade;
	CascadeClassifier face_cascade;
	

	if( !palm_cascade.load( palm_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	double otsu = 0;		
	Scalar mean, stddev;
		
	while(1)
    	{
		Mat frame;
		bool cap_status = cap.read(frame); // get a new frame from camera
		//GaussianBlur(frame, frame ,Size(5,5),0,0);
		if (cap_status == 0)			//If status is false returns 1 to console
		{
			cout << "Error in capturing Frames" << endl;			
			return 1;
		}
		
		Mat img = frame.clone();
				
		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);

		vector<Rect> palms;
		palm_cascade.detectMultiScale( frame_gray, palms, 1.3, 5, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) );
			

		for( size_t i = 0; i < palms.size(); i++ )
		{
			
			rectangle(img, Point(max(palms[i].x - 50, 0), max(palms[i].y - 50, 0)), Point(min(palms[i].x + 50 + palms[i].width, img.cols), min(palms[i].y + 50 + palms[i].height, img.rows)), Scalar(0,255,0), 3);
			Rect padded_palm;
			padded_palm.x = max(palms[i].x - 50, 0);
			padded_palm.y = max(palms[i].y - 50, 0);
			padded_palm.width = min(palms[i].x + 50 + palms[i].width, img.cols) - max(palms[i].x - 50, 0);
			padded_palm.height = min(palms[i].y + 50 + palms[i].height, img.rows) - max(palms[i].y - 50, 0);
			
		  	
			Mat palmROI = frame_gray( padded_palm );
			Mat mask = Mat(Size(palmROI.cols, palmROI.rows), CV_8U, Scalar(0));
			
			Mat rgb_palm_ROI = frame(padded_palm);
			otsu = threshold(palmROI, mask, 0, 255, THRESH_OTSU + THRESH_BINARY_INV);
			
			meanStdDev(rgb_palm_ROI, mean, stddev, mask);
			
		}

		Scalar upper = mean + stddev;
		Scalar lower = mean - stddev;
		
		Mat mask_threshold;
		//double tp = threshold(frame_gray, mask_threshold, otsu, 255, THRESH_BINARY_INV);	
		inRange(frame, lower, upper, mask_threshold);		

		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(10,20));
		dilate(mask_threshold, mask_threshold, element);
		erode(mask_threshold, mask_threshold, element);
		dilate(mask_threshold, mask_threshold, element);
		//erode(mask_threshold, mask_threshold, element);
		

		Mat drawing = Mat::zeros( mask_threshold.size(), CV_8UC3 );
		Mat drawing_1 = Mat::zeros( mask_threshold.size(), CV_8UC3 );			
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours( mask_threshold, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		

		int index = -1;
		double max = -1;
		vector<int>  hull;
		vector<Vec4i> defects;
		for( int i = 0; i < contours.size(); i++ )
		{
			double area = contourArea(contours[i]);
			if (area > max)
			{
				index = i;
				max = area;
			}			 
		}

		if (index > 0)
		{	
			convexHull( contours[index], hull, true, false);
			drawContours( drawing, contours, index, Scalar(255,0,0), 6, 8);
			
			vector<vector<Point>> hull_contours;
			vector<Point> hull_contour;
			for(size_t i = 0; i < hull.size(); i++)
			{
				hull_contour.push_back(contours[index][hull[i]]);
				
			}			
			hull_contours.push_back(hull_contour);
			drawContours( drawing_1,hull_contours, -1, Scalar(0,255,0), 5, 7);
			convexityDefects(contours[index], hull, defects);
			int count_defects = 0;
				
			for(size_t i = 0; i < defects.size(); i++)
			{
				int sx = contours[index][defects[i][0]].x;
				int sy = contours[index][defects[i][0]].y;
				
				int ex = contours[index][defects[i][1]].x;
				int ey = contours[index][defects[i][1]].y;
				
				
				int fx = contours[index][defects[i][2]].x;
				int fy = contours[index][defects[i][2]].y;
				
				double a = (sx - ex) * (sx - ex) + (sy - ey) * (sy - ey);
				double b = (fx - ex) * (fx - ex) + (fy - ey) * (fy - ey);
				double c = (sx - fx) * (sx - fx) + (sy - fy) * (sy - fy);

				double val = (c + b - a) / (2 * sqrt(c) * sqrt(b));
				double angle = acos(val) * 180 / CV_PI;
				
				if (angle <= 90)
				{
					count_defects++;
				} 
				
			}			
			cout << count_defects << endl;
			drawContours( drawing_1, contours, index, Scalar(0,0,255),10, 8);
			if (count_defects == 0){
			putText(img,  "This is 1", Point2f(100,100),   FONT_HERSHEY_SIMPLEX, 3, Scalar(255,0,0)  );}
			if (count_defects == 1){
			putText(img, "This is 2", Point2f(100,100),   FONT_HERSHEY_SIMPLEX, 3, Scalar(255,0,0) );}
			if (count_defects == 2){
			putText(img,  "This is 3", Point2f(100,100),   FONT_HERSHEY_SIMPLEX, 3, Scalar(255,0,0) );}
			if (count_defects == 3){
			putText(img,  "This is 4", Point2f(100,100),   FONT_HERSHEY_SIMPLEX, 3, Scalar(255,0,0)  );}
			if (count_defects >= 4){
			putText(img,  "This is 5", Point2f(100,100),  FONT_HERSHEY_SIMPLEX, 3, Scalar(255,0,0) );}
			
		}
		
		

		/// Show in a window
		namedWindow( "frame", CV_WINDOW_NORMAL );
		//imshow( "Hull demo", drawing );
		imshow( "Hull", drawing_1 );

		
		imshow("threshold", mask_threshold);
			

		
	}

	return 0;
}


