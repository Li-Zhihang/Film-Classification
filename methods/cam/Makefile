
LIBS = -lopencv_core -lopencv_shape -lopencv_videoio -lopencv_features2d -lopencv_highgui -lopencv_calib3d -lopencv_imgproc -lopencv_xfeatures2d
CPP_FLAG = -I/usr/local/include/opencv4
cameraMotionDesc: cameraMotionDesc.cpp
	g++ -O3  $^ -o $@  $(LIBS) $(CPP_FLAG)