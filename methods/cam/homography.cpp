#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


#include <vector>
#include <stdio.h>
#include <iostream>

#define MAX_PATH_LEN 1024
#define H_NUM_MATCHES 5
#define H_ERROR_THRESHOLD 256
#define EXTN_LIECOEFF ".lie"
#define EXTN_STATFEAT ".stat"


struct params {
    /* Processed output to be used as input by other modules */
    cv::VideoCapture video_src;
    int frame_width;
    int frame_height;
    int video_fps;
    int num_frames;
    int window_size;

    char output_suffix[MAX_PATH_LEN];
    char homography_file[MAX_PATH_LEN];
    char liespace_rep_file[MAX_PATH_LEN];
    char ts_feat_file[MAX_PATH_LEN];
    char stat_feat_file[MAX_PATH_LEN];
};
typedef struct params P_PARAMS;


void get_homography_sequence(cv::VideoCapture capture, std::vector<cv::Mat> &vec_H);
void construct_timeseries(const std::vector<cv::Mat> &vec_L, std::vector<cv::Mat> &vec_LTS);

double entropy(const cv::Mat &data);
int zero_crossing(const cv::Mat &data);
double corr(const cv::Mat &x, const cv::Mat &y);
cv::Mat compute_stat_feat(const std::vector<cv::Mat> &vec_LTS);

/******************************************************************/
/* BEGIN Program specific Utility,I/O routines for debug purpose  */
/******************************************************************/
void print_matrix(const cv::Mat &x, const char *marker) {
    fprintf(stderr, "%dx%d matrix: %s\n", x.rows, x.cols, marker);
    for (int icnt = 0; icnt < x.rows; icnt++) {
        for (int jcnt = 0; jcnt < x.cols; jcnt++)
            fprintf(stderr, "%2.5lf ", x.at<double>(icnt, jcnt));
        fprintf(stderr, "\n");
    }
}
void save_matrix(const cv::Mat &x, const char *filename) {
    FILE *fp;
    fprintf(stderr, "%dx%d matrix\n", x.rows, x.cols);
    fp = fopen(filename, "w+");
    for (int icnt = 0; icnt < x.rows; icnt++) {
        for (int jcnt = 0; jcnt < x.cols; jcnt++)
            fprintf(fp, "%2.5lf ", x.at<double>(icnt, jcnt));
        fprintf(stderr, "\n");
    }
    fclose(fp);
}

void print_vector_matrix(std::vector<cv::Mat> &x, const char *marker) {
    while (!x.empty()) {
        print_matrix(x.back(), marker);
        x.pop_back();
    }
}
void save_vectors(const std::vector<cv::Mat> &x, char *fname) {
    FILE *fp;
    int e_count;
    fp = fopen(fname, "w+");
    for (int v_count = 0; v_count < x.size(); v_count++) {
        for (e_count = 0; e_count < x[v_count].cols - 1; e_count++)
            fprintf(fp, "%2.5lf ", x[v_count].at<double>(0, e_count));
        /* Print last element in the vector - to avoid trailing space */
        fprintf(fp, "%2.5lf\n", x[v_count].at<double>(0, e_count));
    }
    fclose(fp);
}

void usage(char **argv) {
    std::cout << "Usage: " << argv[0] << "\n"
                                    "         -i /video/file/path\n"
                                    "         -w window size\n"
                                    "         -s output/suffix/file/path\n"
         << std::endl;
}

/* Parse command line option */
int parse_options(int argc, char **argv, P_PARAMS &p) {
    int i;

    if (argc == 1) {
        usage(argv);
        return 1;
    }
    for (i = 1; i < argc; i++) {
        /* Input video file */
        if (!strcmp(argv[i], "-i")) {
            std::string str = std::string(argv[++i]);
            p.video_src.open(str);
            if (!p.video_src.isOpened()) {
                std::cerr << "ERR: Failed to open a video file: \n"
                     << std::endl;
                usage(argv);
                return 1;
            }
            /* Perform populating the structure with video information */
            p.frame_width = p.video_src.get(cv::CAP_PROP_FRAME_WIDTH);
            p.frame_height = p.video_src.get(cv::CAP_PROP_FRAME_HEIGHT);
            p.video_fps = p.video_src.get(cv::CAP_PROP_FPS);
            p.num_frames = p.video_src.get(cv::CAP_PROP_FRAME_COUNT);
        }

        /* window size to chose from (default 1) */
        if (!strcmp(argv[i], "-w")) {
            p.window_size = strtod(argv[++i], NULL);
        }

        /* This is used as a base name, different extensions are added for 
    different output files */
        if (!strcmp(argv[i], "-s")) {
            strcpy(p.output_suffix, argv[++i]);
            strcpy(p.liespace_rep_file, p.output_suffix);
            strcat(p.liespace_rep_file, EXTN_LIECOEFF);

            strcpy(p.stat_feat_file, p.output_suffix);
            strcat(p.stat_feat_file, EXTN_STATFEAT);
        }
    }
    fprintf(stdout, "INF:Using \n");

    fprintf(stdout, "  Video resolution        : %d x %d\n"
                    "  Video Frames per second : %d\n"
                    "  Number of frames        : %d\n"
                    "  Homography Window Size  : %d\n"
                    "  Output Lie Space Rep    : %s\n"
                    "  Output Stat. Feat. file : %s\n",
            p.frame_width, p.frame_height, p.video_fps, p.num_frames, p.window_size,
            p.liespace_rep_file, p.stat_feat_file);
    return 0;
}

/* Core function responsible for video processing 
 Control Flow:
   For each pair of frames 
   Compute SIFT/SURF features
   Get corresponding x,y co-ordinates for homography computation
   Compute homography
   Assign  current features to previous frame features (saves re-computation)
   Store homographies in a std::vector
   Compute Statistical features
   Store Features in a std::vector
   Output data
*/
int process(P_PARAMS &p) {
    std::vector<cv::Mat> vec_H, vec_L, vec_LTS;
    cv::Mat stat_feat;
    fprintf(stderr, "INFO: Computing Homographies.\n");
    get_homography_sequence(p.video_src, vec_H);
    /* Computing statistical features */
    fprintf(stderr, "INFO: Computing Statisitical Features.\n");
    construct_timeseries(vec_H, vec_LTS);
    stat_feat = compute_stat_feat(vec_LTS);
    save_matrix(stat_feat, p.stat_feat_file);

    return 0;
}

/******************************************************************/
/* BEGIN Program specific Core routines                           */
/******************************************************************/

/* 
  Given source and destination cv::keypoints from two subsequent images, 
  find corresponding points using standard descriptor matching 
  using correspondence, compute homography 
  return homography matrix elements in the form of 1x8 vector.
  Generate a sequence of subsequent Homography elements
*/
/* Helper routines for Homography computation */
void drawMatchesRelative(const std::vector<cv::KeyPoint> &train, const std::vector<cv::KeyPoint> &query,
                         const std::vector<cv::DMatch> &matches, const cv::Mat &img, const std::vector<unsigned char> &mask = std::vector<unsigned char>()) {
    for (int i = 0; i < static_cast<int>(matches.size()); i++) {
        if (mask.empty() || mask[i]) {
            cv::Point2f pt_new = query[matches[i].queryIdx].pt;
            cv::Point2f pt_old = train[matches[i].trainIdx].pt;
            cv::Point2f dist = pt_new - pt_old;

            cv::line(img, pt_new, pt_old, cv::Scalar(125, 255, 125), 1);
            cv::circle(img, pt_new, 2, cv::Scalar(255, 0, 125), 1);
        }
    }
}
// Takes a descriptor and turns it into an xy point
void keypoints2points(const std::vector<cv::KeyPoint> &in, std::vector<cv::Point2f> &out) {
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        out.push_back(in[i].pt);
}

// Takes an xy point and appends that to a cv::keypoint structure
void points2keypoints(const std::vector<cv::Point2f> &in, std::vector<cv::KeyPoint> &out) {
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        out.push_back(cv::KeyPoint(in[i], 1));
}

// Uses computed homography H to warp original input points to new planar position
void warpKeypoints(const cv::Mat &H, const std::vector<cv::KeyPoint> &in, std::vector<cv::KeyPoint> &out) {
    std::vector<cv::Point2f> pts;
    keypoints2points(in, pts);
    std::vector<cv::Point2f> pts_w(pts.size());
    cv::Mat m_pts_w(pts_w);
    perspectiveTransform(cv::Mat(pts), m_pts_w, H);
    points2keypoints(pts_w, out);
}

// Converts matching indices to xy points
void matches2points(const std::vector<cv::KeyPoint> &train, const std::vector<cv::KeyPoint> &query,
                    const std::vector<cv::DMatch> &matches, std::vector<cv::Point2f> &pts_train, std::vector<cv::Point2f> &pts_query) {
    pts_train.clear();
    pts_query.clear();
    pts_train.reserve(matches.size());
    pts_query.reserve(matches.size());

    size_t i = 0;
    for (; i < matches.size(); i++) {
        const cv::DMatch &dmatch = matches[i];
        pts_query.push_back(query[dmatch.queryIdx].pt);
        pts_train.push_back(train[dmatch.trainIdx].pt);
    }
}

void resetH(cv::Mat &H) {
    H = cv::Mat::eye(3, 3, CV_64F);
}

cv::Mat windowedMatchingMask(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, float maxDeltaX, float maxDeltaY) {
    if (keypoints1.empty() || keypoints2.empty())
        return cv::Mat();

    int n1 = static_cast<int>(keypoints1.size()), n2 = static_cast<int>(keypoints2.size());
    cv::Mat mask(n1, n2, CV_8UC1);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            cv::Point2f diff = keypoints2[j].pt - keypoints1[i].pt;
            mask.at<uchar>(i, j) = std::abs(diff.x) < maxDeltaX && std::abs(diff.y) < maxDeltaY;
        }
    }
  return mask;
}


void get_homography_sequence(cv::VideoCapture capture, std::vector<cv::Mat> &vec_H) {
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
    cv::Mat frame;

    std::vector<cv::DMatch> matches;

    cv::BFMatcher desc_matcher(cv::NORM_HAMMING);

    std::vector<cv::Point2f> train_pts, query_pts;
    std::vector<cv::KeyPoint> train_kpts, query_kpts;
    std::vector<unsigned char> match_mask;

    cv::Mat gray;
    cv::Mat train_desc, query_desc;
    const int DESIRED_FTRS = 500;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(10, true);
    cv::Mat H;
    cv::Mat H_prev = cv::Mat::eye(3, 3, CV_64F);
    for (;;) {
        capture >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        // Find interest points
        detector->detect(gray, query_kpts);
        // Compute brief descriptors at each cv::keypoint location
        brief->compute(gray, query_kpts, query_desc);

        if (!train_kpts.empty()) {
            std::vector<cv::KeyPoint> test_kpts;
            warpKeypoints(H_prev.inv(), query_kpts, test_kpts);

            cv::Mat mask = windowedMatchingMask(test_kpts, train_kpts, 25, 25);
            desc_matcher.match(query_desc, train_desc, matches, mask);

            drawKeypoints(frame, test_kpts, frame, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

            matches2points(train_kpts, query_kpts, matches, train_pts, query_pts);

            if (matches.size() > 5) {
                H = cv::findHomography(train_pts, query_pts, cv::RANSAC, 4, match_mask);
                if (cv::countNonZero(cv::Mat(match_mask)) > 15) {
                    H_prev = H;
                } else {
                    resetH(H_prev);
                    // drawMatchesRelative(train_kpts, query_kpts, matches, frame, match_mask);
                }
            } else {
                resetH(H_prev);
            }
        } else {
            H_prev = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat out;
            // drawcv::Keypoints(gray, query_kpts, out);
            frame = out;
        }
        vec_H.push_back(H_prev.reshape(0, 9).t());
        // imshow("frame", frame);
        // waitKey(2);
        train_kpts = query_kpts;
        query_desc.copyTo(train_desc);
    }
}


void construct_timeseries(const std::vector<cv::Mat> &vec_L, std::vector<cv::Mat> &vec_LTS) {
    int n = vec_L.size();
    int dim = vec_L.back().cols;
    /* column matrix to store time series */
    cv::Mat x = cv::Mat::zeros(n, 1, vec_L.back().type());
    /* reverse iteration to make sure that the first coefficient time series 
  comes as first element in the vector vec_LTS() */
    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < n; i++)
            x.at<double>(i, 0) = vec_L[i].at<double>(0, j);
        /* put matrix in vector - the transpose operation copies the data */
        vec_LTS.push_back(x.t());
    }
    x.release();
}
/* Compute statistical features from time series data. Following are the 
statistical features that are computed. 
For each dimension: 
  mean
  variance
  mean crossing rate [length(crossing(sig-mean(sig)))/length(sig)]
  average root mean square
  mean of skew [mean(((sig-mean(sig)).^3)./(std(sig)^3))]
  variance of skew [mean(((sig-mean(sig)).^3)./(std(sig)^3))]
  mean of kurtosis [mean((sig-mean(sig)).^4)/(std(sig)^4);]
  variance of kurtosis [mean((sig-mean(sig)).^4)/(std(sig)^4);]

For each dimension pair:
  correlation (7+6+5+4+3+2+1)=28

Total: 8x8 + 28 = 92 Dimensional feature std::vector
*/
cv::Mat compute_stat_feat(const std::vector<cv::Mat> &vec_LTS) {
    int j = 0, sig_length;
    cv::Scalar mu, sigma, s_mu, s_sigma, k_mu, k_sigma;
    cv::MatND mask;
    cv::Mat ms, res;
    int dim = vec_LTS.size();
    cv::Mat stat_feat = cv::Mat::zeros(1, 92, vec_LTS.back().type());

    /* For independent features */
    for (int i = 0; i < dim - 1; i++) {
        /* mean and variance */
        meanStdDev(vec_LTS[i], mu, sigma, mask);
        stat_feat.at<double>(0, j++) = mu.val[0];
        stat_feat.at<double>(0, j++) = pow(sigma.val[0], 2);

        /* Normalized crossing rate */
        cv::Mat ms = vec_LTS[i] - mu.val[0];
        sig_length = vec_LTS[i].cols;
        stat_feat.at<double>(0, j++) = static_cast<double>(zero_crossing(ms) / sig_length);
        /* Normalized RMS */
        stat_feat.at<double>(0, j++) = static_cast<double>(sqrt(pow(norm(vec_LTS[i], cv::NORM_L2), 2) / sig_length));
        /* Computing skew */
        pow(ms, 3, res);
        res = res / pow(static_cast<double>(sigma.val[0]), 3);
        meanStdDev(res, s_mu, s_sigma, mask);
        stat_feat.at<double>(0, j++) = s_mu.val[0];
        stat_feat.at<double>(0, j++) = s_sigma.val[0];
        /* Entropy */
        // stat_feat.at<double>(0, j++) = entropy(vec_LTS[i]);
        /* Kurtosis */
        pow(ms, 4, res);
        res = res / pow(sigma.val[0], 4);
        meanStdDev(res, k_mu, k_sigma, mask);
        stat_feat.at<double>(0, j++) = k_mu.val[0];
        stat_feat.at<double>(0, j++) = k_sigma.val[0];
    }
    /* For correlation features */
    cv::Mat flag = cv::Mat::zeros(dim - 1, dim - 1, CV_32FC1);

    for (int i = 0; i < dim - 1; i++)
        for (int k = 0; k < dim - 1; k++)
            if (i != k && flag.at<float>(i, k) == 0 && flag.at<float>(k, i) == 0) {
                stat_feat.at<double>(0, j++) = corr(vec_LTS[i], vec_LTS[k]);
                flag.at<float>(i, k) = 1;
                flag.at<float>(k, i) = 1;
            }
    // print_matrix (stat_feat, "Rows of stat_feat");
    return (stat_feat);
}
double entropy(const cv::Mat &data) {
    int i;

    double dE = 0.0, d, eps = 0.00000000001;
    for (i = 0; i < data.cols; i++) {
        d = data.at<double>(0, i);
        dE = dE - d * log(d + eps) - (1.0 - d) * log(1.0 - d + eps);
    }
    return dE;
}

int zero_crossing(const cv::Mat &data) {
    int i, j = 0;
    int sign1, sign2;

    for (i = 0; i < data.cols - 1; i++) {
        sign1 = (data.at<double>(0, i) > 0) ? 1 : 0;
        sign2 = (data.at<double>(0, i + 1) > 0) ? 1 : 0;
        if (sign1 != sign2)
            j++;
    }
    return j;
}
double corr(const cv::Mat &x, const cv::Mat &y) {
    double var_x, var_y, s = 0, s_x = 0, s_y = 0;
    cv::Scalar mu_x, mu_y, sigma_x, sigma_y;
    cv::MatND mask;

    meanStdDev(x, mu_x, sigma_x, mask);
    meanStdDev(y, mu_y, sigma_y, mask);

    for (int i = 0; i < x.cols; i++) {
        s = s + (x.at<double>(0, i) - mu_x.val[0]) * (y.at<double>(0, i) - mu_y.val[0]);
        s_x = s_x + pow(x.at<double>(0, i) - mu_x.val[0], 2);
        s_y = s_y + pow(y.at<double>(0, i) - mu_y.val[0], 2);
    }
    return static_cast<double>(s) / (sqrt(s_x) * sqrt(s_y));
}
/******************************************************************/
/* END Program specific Core routines                           */
/******************************************************************/

int main(int argc, char **argv) {
    P_PARAMS p;
    if (parse_options(argc, argv, p) == 0) {
        return process(p);
    }
}
