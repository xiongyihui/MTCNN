#include "mtcnn.h"

#include <unistd.h>

#include "timing.h"

#include "opencv2/opencv.hpp"


using namespace cv;



int main(int argc, char **argv) {
	printf("mtcnn face detection\n");

	if (argc < 2) {
		printf("usage: %s  model_path\n ", argv[0]);
		printf("eg: %s  ../models\n ", argv[0]);
		return 0;
	}
	const char *model_path = argv[1];

	
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "video is not open" << endl;
        return -1;
    }


    Mat frame;
	ncnn::Mat ncnn_img;
	std::vector<Bbox> finalBbox;
	MTCNN mtcnn(model_path);

    mtcnn.SetMinFace(60);

    while (1) {
        cap >> frame;

        ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows);
        double startTime = now();
        mtcnn.detect(ncnn_img, finalBbox);
        double nDetectTime = calcElapsed(startTime, now());
        printf("time: %d ms.\n ", (int)(nDetectTime * 1000));
        int num_box = finalBbox.size();
        printf("face num: %u \n", num_box);

        for (int i = 0; i < num_box; i++) {
            // Plot bounding box
            rectangle(frame, Point(finalBbox[i].x1, finalBbox[i].y1), 
                Point(finalBbox[i].x2, finalBbox[i].y2), Scalar(0, 0, 255), 2, 8, 0);

            // Plot facial landmark
            for (int num = 0; num < 5; num++)
            {
                circle(frame, Point(finalBbox[i].ppoint[num], finalBbox[i].ppoint[num + 5]), 3, Scalar(0, 255, 255), -1);
            }

        }

        imshow("frame", frame);
        if (waitKey(1) == 'q')
            break;
    }


    return 0;
}