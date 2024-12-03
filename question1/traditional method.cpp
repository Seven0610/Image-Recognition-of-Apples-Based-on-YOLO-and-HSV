void Apple::Detect_Apple(cv::Mat& src)
{
	
	Mat img = imread("D:/VSprojection/detect_apple/detect_apple/1.jpeg", cv::IMREAD_COLOR);//用来输出
	//step1:BGR->HSV
	Mat src_HSV;
	cvtColor(src, src_HSV, COLOR_BGR2HSV);
	//imshow("source_image", src);
	medianBlur(src_HSV, src_HSV, 5);//median filtering

	//step2:Extract apple
	int imgrow = src.rows;
	int imgcol = src.cols;
	for (int m = 0; m < imgrow; m++)
	{
		for (int n = 0; n < imgcol; n++)
		{
			//Extract red areas
			if (!((((src_HSV.at<Vec3b>(m, n)[0] >= 0) && (src_HSV.at<Vec3b>(m, n)[0] <= 15)) ||
				(src_HSV.at<Vec3b>(m, n)[0] >= 125) && (src_HSV.at<Vec3b>(m, n)[0] <= 180)) && (src_HSV.at<Vec3b>(m, n)[2] >= 46) &&
				(src_HSV.at<Vec3b>(m, n)[1] >= 43)))

			{
				//if ((src_HSV.at<Vec3b>(m, n)[0] >= 35 && src_HSV.at<Vec3b>(m, n)[0] <= 77))
				//{
				src.at<Vec3b>(m, n)[0] = 255;
				src.at<Vec3b>(m, n)[1] = 255;
				src.at<Vec3b>(m, n)[2] = 255;
				//}
			}
		}
	}
	Mat outimage = src.clone();
	//imshow("hongse", outimage);
	//step3: A binarization is carried out to facilitate the extraction of contour
	cvtColor(outimage, outimage, COLOR_BGR2GRAY);
	threshold(outimage, outimage, 300, 255, CV_THRESH_OTSU | CV_THRESH_BINARY_INV);

	//step4:An open operation is performed using morphological operations
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	morphologyEx(outimage, outimage, MORPH_OPEN, element);

	//step5: Detection profile
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(outimage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//step6: Set an area threshold and box anything larger than the threshold
	int min_val = 100;
	//const char* name = "123456";
	int k = 6;
	string str;
	set<Apple, compareApple>st;
	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		if (area > min_val)
		{

			Rect rect = boundingRect(contours[i]);
			Apple app(rect, area);
			st.insert(app);
			//rectangle(src, rect, Scalar(255, 255, 0), 2);
		}
	}
	//cout << st.size() << endl;
	for (set<Apple, compareApple>::iterator it = st.begin(); it != st.end(); it++)
	{
		rectangle(img, it->m_rect, Scalar(255, 255, 0), 2);
		str = to_string(k--) + "apple";
		putText(img, str, Point(it->m_rect.x, it->m_rect.y + 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(0, 0, 255), 2);
	}
	imshow("outimage", img);
}
