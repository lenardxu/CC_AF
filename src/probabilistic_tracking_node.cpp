#include "probabilistic_tracking_node.hpp"

#include <string>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>

#define STATE_LANE_LEFT 1
#define STATE_LANE_RIGHT 0

#define TYPE_RIGHT 0
#define TYPE_LEFT 1
#define TYPE_CENTER_DASHED 2
#define TYPE_CENTER_DOUBLE 3
#define TYPE_CENTER_COMBINED 4
#define TYPE_PARKING_PERPENDICULAR 5
#define TYPE_PARKING_PARALLEL 6
#define TYPE_HALT_STOP 7
#define TYPE_HALT_LOOK 8
#define TYPE_CROSSING 9
#define TYPE_NUMBER_ARROW 10
#define TYPE_RESTRICTED_AREA 11

ProbabilisticTrackingNode::ProbabilisticTrackingNode(std::string name, bool publishImageOutput, bool testMode)
: libpsaf::LaneDetectionInterface(name,
    1, std::vector<std::string>(
      {"color/image_raw"}), "tracking_node/tracking", "state_machine/state",
    "lane_detection_interface/image", "rgb8", "lane_detection_interface/lane_marking", rclcpp::SensorDataQoS())
{
  if (testMode) {
    std::cout << "test is on!"<< std::endl;
    this->publishImageOutput = publishImageOutput;
    lane_state = STATE_LANE_RIGHT;
  } else {
    const rclcpp::SensorDataQoS QOS{rclcpp::KeepLast(1)};
    this->publishImageOutput = publishImageOutput;

    /*
    cameraImageSubscriber = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/color/image_raw", QOS,
      std::bind(
        &SlidingWindowNode::colorImageReceivedCallback,
        this, std::placeholders::_1));
    */

    stateSubscriber_ = this->create_subscription<std_msgs::msg::Int64>(
            "state_machine/state", 10,
            std::bind(&ProbabilisticTrackingNode::stateReceivedCallback, this, std::placeholders::_1));

    resultImagePublisher = this->create_publisher<sensor_msgs::msg::Image>(
            "/lane_detection/output_image", QOS);
    currentAnglePublisher = this->create_publisher<std_msgs::msg::Float32>(
            "/lane_detection/current_angle", QOS);
    targetAnglePublisher = this->create_publisher<sensor_msgs::msg::Range>(
            "/lane_detection/target_angle", QOS);

    this->declare_parameter("homography"); /* all ros2 parameters should first be declared **/
    this->declare_parameter("square_width", 0.04f); /* all ros2 parameters should first be declared **/
    this->declare_parameter("nr_particle", 200); /* all ros2 parameters should first be declared **/
    this->declare_parameter("meter_to_pixel_ratio", 3137.7);
    this->get_parameter("nr_particle", nr_particle);
    this->get_parameter("square_width", sqaure_width);
    this->get_parameter("meter_to_pixel_ratio", meter_to_pixel_ratio);
    scale = meter_to_pixel_ratio; //replace the math. relationship between 100 pixels and square width with meter_to_pixel_ratio
    // scale = 100.0f / scale;

    // lane_state = STATE_LANE_RIGHT; // to be removed if the state is being derived normally from state machine
    // default value
    float warpm[9] = {-2.91334138e+00, -8.53644184e+00, 2.49017830e+03, 1.41475922e-01, -2.09373084e+01, 5.04251409e+03, 4.87848005e-05, -5.43090446e-03, 1.00000000e+00};
    rclcpp::Parameter double_array_param = this->get_parameter("homography");
    std::vector<double> values = double_array_param.as_double_array();
    if (values.size() == 9) {
      for (int i = 0; i < 9; ++i) {
        warpm[i] = values.at(i);
      }
    }
    warpmat = cv::Mat(3, 3, CV_32F, warpm);
  }
}

ProbabilisticTrackingNode::~ProbabilisticTrackingNode()
{
    bottom_windows_.clear();
    bottom_windows_.shrink_to_fit();

    right_windows_.clear();
    right_windows_.shrink_to_fit();

    left_windows_.clear();
    left_windows_.shrink_to_fit();
}

void ProbabilisticTrackingNode::stateReceivedCallback(const std_msgs::msg::Int64::SharedPtr msg)
{
    ProbabilisticTrackingNode::state = msg->data;
    RCLCPP_INFO(get_logger(), "Received new state: %f", ProbabilisticTrackingNode::state);
}

void ProbabilisticTrackingNode::processImage(cv::Mat & img, int sensor)
{
  cv::Mat imgCandidatesVertical = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8U);
  cv::Mat imgCandidatesHorizontal = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8U);
  cv::Mat imgGray;
  cv::Mat imgGradientHorizontal;
  cv::Mat imgGradientVertical;
  auto start_time = std::chrono::high_resolution_clock::now();
  cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
  auto gray_time = std::chrono::high_resolution_clock::now();
  dynamicHorizontalGradient(imgGray, imgGradientHorizontal);
  verticalGradient(imgGray, imgGradientVertical);
  auto gradient_time = std::chrono::high_resolution_clock::now();
  std::vector<cv::Point2f> vecHorizontal;
  std::vector<cv::Point2f> vecVertical;
  findMatchingGradientsHorizontal(imgGradientHorizontal, imgCandidatesHorizontal, vecHorizontal);
  // std::cout << "size horizontal " << vecHorizontal.size() << std::endl;
  findMatchingGradientsVertical(imgGradientVertical, imgGradientHorizontal, imgCandidatesVertical, vecVertical);
  auto matched_time = std::chrono::high_resolution_clock::now();
  std::vector<cv::Point2f> pointsVertical;
  std::vector<cv::Point2f> pointsHorizontal;
  transformPoints(vecVertical, pointsVertical, warpmat);
  transformPoints(vecHorizontal, pointsHorizontal, warpmat);
  auto transformed_time = std::chrono::high_resolution_clock::now();
  cv::Mat imgCandidatesTransformed = cv::Mat::zeros(cv::Size(3000, 3500), CV_8U);
  cv::Mat params = findLaneBoundary(imgCandidatesHorizontal, imgCandidatesTransformed, pointsHorizontal);
  auto find_boundary_time = std::chrono::high_resolution_clock::now();
  std::vector<std::pair<int, std::vector<cv::Point2f>>> markings(3);
  classifyAccordingToRulesAdaptive(pointsHorizontal, pointsVertical, markings, params);
  // classifyAccordingToRules(pointsHorizontal, pointsVertical, img, markings); /**< deprecated for now */
  auto classify_time = std::chrono::high_resolution_clock::now();

  double angle;
  if (getCourseAngle(markings, angle, imgCandidatesTransformed) != 0){
      return;
  }
  // std::vector<libpsaf_msgs::msg::LaneMarkings> markingsMsgs; /**< deprecated for now since we want the current angle as output message */
  // markingsToMessages(markings, markingsMsgs); /**< deprecated for now since we want the current angle as output message */

  /* for testing the markingsMsgs after transformation above **/
  // test_lane_markings = markingsMsgs;
  //printTransformed(pointsVertical, pointsHorizontal, imgCandidatesTransformed);

  auto get_angle_time = std::chrono::high_resolution_clock::now();
  // cv::Mat target = cv::Mat::zeros(cv::Size(3000, 3500), CV_8U);
  // cv::cvtColor(target, target, cv::COLOR_GRAY2BGR);
  printMarkings(markings, imgCandidatesTransformed);
  auto printed_time = std::chrono::high_resolution_clock::now();
  //condense.filter(0, 0, vec);
  auto filtered_time = std::chrono::high_resolution_clock::now();
  //printParticles(condense.getParticles(), img);
  assumeLaneState(markings, lane_state);
  auto assume_time = std::chrono::high_resolution_clock::now();

  auto duration_gray = std::chrono::duration_cast<std::chrono::milliseconds>(gray_time - start_time);
  auto duration_gradient = std::chrono::duration_cast<std::chrono::milliseconds>(gradient_time - gray_time);
  auto duration_match = std::chrono::duration_cast<std::chrono::milliseconds>(matched_time - gradient_time);
  auto duration_transform = std::chrono::duration_cast<std::chrono::milliseconds>(transformed_time - matched_time);
  auto duration_find_boundary = std::chrono::duration_cast<std::chrono::milliseconds>(find_boundary_time - transformed_time);
  auto duration_classify = std::chrono::duration_cast<std::chrono::milliseconds>(classify_time - find_boundary_time);
  auto duration_get_angle = std::chrono::duration_cast<std::chrono::milliseconds>(get_angle_time - classify_time);
  auto duration_print = std::chrono::duration_cast<std::chrono::milliseconds>(printed_time - get_angle_time);
  auto duration_filter = std::chrono::duration_cast<std::chrono::milliseconds>(filtered_time - printed_time);
  auto duration_assume = std::chrono::duration_cast<std::chrono::milliseconds>(assume_time - filtered_time);
  auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(filtered_time - start_time);
  RCLCPP_INFO(this->get_logger(),
              "gray %d, gradient %d, match %d, transform %d, boundary %d, classify %d, angle %d, print %d, filter %d, assume %d, all %d",
              duration_gray.count(), duration_gradient.count(), duration_match.count(), duration_transform.count(),
              duration_find_boundary.count(), duration_classify.count(), duration_get_angle.count(), duration_print.count(),
              duration_filter.count(), duration_assume.count(), duration_all.count());
  //cv::cvtColor(imgCandidatesTransformed, img, cv::COLOR_GRAY2BGR);

  std_msgs::msg::Float32 cAngleMsg;
  cAngleMsg.data = -angle;
  currentAnglePublisher->publish(cAngleMsg);

  sensor_msgs::msg::Range tAngleMsg;
  tAngleMsg.header.stamp = now();
  tAngleMsg.range = 0.0;
  targetAnglePublisher->publish(tAngleMsg);

  if (publishImageOutput) {
      // publishImage(target); /**< deprecated since we need the explicit output of corresponding topic */
      cv_bridge::CvImagePtr cv_ptr;
      cv_ptr->image = imgCandidatesTransformed;
      cv_ptr->encoding = sensor_msgs::image_encodings::RGB8;
      resultImagePublisher->publish(*(cv_ptr->toImageMsg()));
  }

  /* // Instead of publishing the fitted model parameters and distance between center point of each lane markings and
   * // (1500,3000), we publish current angle as desired.
  for (int i = 0; i < markingsMsgs.size(); ++i) {
    publishLaneMarkings(markingsMsgs.at(i));
  } */


}

void ProbabilisticTrackingNode::processImageForTest(cv::Mat & img, const std::string & savePath, bool imgProcFlag, cv::VideoWriter& video)
{
  cv::Mat imgCandidatesVertical = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8U);
  cv::Mat imgCandidatesHorizontal = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8U);
  cv::Mat imgGray;
  cv::Mat imgGradientHorizontal;
  cv::Mat imgGradientVertical;
  auto start_time = std::chrono::high_resolution_clock::now();
  cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
  auto gray_time = std::chrono::high_resolution_clock::now();
  dynamicHorizontalGradient(imgGray, imgGradientHorizontal);
  verticalGradient(imgGray, imgGradientVertical);
  auto gradient_time = std::chrono::high_resolution_clock::now();

  /* //for testing
  cv::Mat c_gradient;
  cv::hconcat(imgGradientHorizontal, imgGradientVertical, c_gradient);
  bool check_1 = cv::imwrite(std::filesystem::path(savePath).parent_path() / "concatnated_gradients.png", c_gradient);
  if (!check_1) {
      std::cout << "Mission - Saving the image, FAILED" << std::endl;
      return;}
  */

  std::vector<cv::Point2f> vecHorizontal;
  std::vector<cv::Point2f> vecVertical;
  findMatchingGradientsHorizontal(imgGradientHorizontal, imgCandidatesHorizontal, vecHorizontal);
  std::cout << "size horizontal " << vecHorizontal.size() << std::endl;
  findMatchingGradientsVertical(imgGradientVertical, imgGradientHorizontal, imgCandidatesVertical, vecVertical);
  auto matched_time = std::chrono::high_resolution_clock::now();

  /* //for testing
  cv::Mat c_candidate;
  cv::hconcat(imgCandidatesHorizontal, imgCandidatesVertical, c_candidate);
  bool check_2 = cv::imwrite(std::filesystem::path(savePath).parent_path() / "concatnated_candidates.png", c_candidate);
  if (!check_2) {
      std::cout << "Mission - Saving the image, FAILED" << std::endl;
      return;}
  */

  std::vector<cv::Point2f> pointsVertical;
  std::vector<cv::Point2f> pointsHorizontal;
  transformPoints(vecVertical, pointsVertical, warpmat);
  transformPoints(vecHorizontal, pointsHorizontal, warpmat);
  auto transformed_time = std::chrono::high_resolution_clock::now();
  cv::Mat imgCandidatesTransformed = cv::Mat::zeros(cv::Size(3000, 3500), CV_8U);
  cv::Mat params = findLaneBoundaryForTest(imgCandidatesHorizontal, imgCandidatesTransformed, pointsHorizontal, savePath);
  std::cout << "boundary is found, with params being " << "\n" << params << std::endl;
  auto find_rightboundary_time = std::chrono::high_resolution_clock::now();

    /* //  for testing purpose in terms of printed transformed edge points
    cv::Mat imgCandidatesTransformedVisualize = cv::Mat::zeros(cv::Size(3000, 3500), CV_8U);
    cv::cvtColor(imgCandidatesTransformedVisualize, imgCandidatesTransformedVisualize, cv::COLOR_GRAY2BGR);
    printTransformed(pointsVertical, pointsHorizontal, imgCandidatesTransformedVisualize);
    bool check_3 = cv::imwrite(std::filesystem::path(savePath).parent_path() / "transformed_candidates.png", imgCandidatesTransformedVisualize);
    if (!check_3) {
        std::cout << "Mission - Saving the image, FAILED" << std::endl;
        return;}
    */

  // the removal function works when the functional driving scenarios are not included, e.g., parallel parking markings
  // removeOutlierFromVecHorizontal(pointsHorizontal, params);
  std::vector<std::pair<int, std::vector<cv::Point2f>>> markings(3);
  classifyAccordingToRulesAdaptive(pointsHorizontal, pointsVertical, markings, params);
  std::cout << "classification is of lane markings is done."  << std::endl;
  auto classify_time = std::chrono::high_resolution_clock::now();
  double angle;
  if (getCourseAngle(markings, angle, imgCandidatesTransformed) != 0){
      std::cerr << "angle cannot be solved due to lack of points." << std::endl;
      return;
  }
  auto get_angle_time = std::chrono::high_resolution_clock::now();
  // cv::Mat target = cv::Mat::zeros(cv::Size(3000, 3500), CV_8U);
  // cv::cvtColor(target, target, cv::COLOR_GRAY2BGR);
  printMarkingsForTest(markings, imgCandidatesTransformed, savePath, imgProcFlag, video);
  auto printed_time = std::chrono::high_resolution_clock::now();
  //condense.filter(0, 0, vec);
  auto filtered_time = std::chrono::high_resolution_clock::now();
  //printParticles(condense.getParticles(), img);
  assumeLaneState(markings, lane_state);
  auto assume_time = std::chrono::high_resolution_clock::now();

  auto duration_gray = std::chrono::duration_cast<std::chrono::milliseconds>(gray_time - start_time);
  auto duration_gradient = std::chrono::duration_cast<std::chrono::milliseconds>(gradient_time - gray_time);
  auto duration_match = std::chrono::duration_cast<std::chrono::milliseconds>(matched_time - gradient_time);
  auto duration_transform = std::chrono::duration_cast<std::chrono::milliseconds>(transformed_time - matched_time);
  auto duration_find_rightboundary = std::chrono::duration_cast<std::chrono::milliseconds>(find_rightboundary_time - transformed_time);
  auto duration_classify = std::chrono::duration_cast<std::chrono::milliseconds>(classify_time - find_rightboundary_time);
  auto duration_get_angle = std::chrono::duration_cast<std::chrono::milliseconds>(get_angle_time - classify_time);
  auto duration_print = std::chrono::duration_cast<std::chrono::milliseconds>(printed_time - get_angle_time);
  auto duration_filter = std::chrono::duration_cast<std::chrono::milliseconds>(filtered_time - printed_time);
  auto duration_assume = std::chrono::duration_cast<std::chrono::milliseconds>(assume_time - filtered_time);
  auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(filtered_time - start_time);
  std::cout << "finding boundary takes " << duration_find_rightboundary.count() << " ms.\n"
            << "calculating angle takes" << duration_get_angle.count() << " ms.\n"
            << "assuming takes" << duration_assume.count() << " ms.\n"
            << "image processing takes " << duration_all.count() << " ms." << std::endl;
}

cv::Mat ProbabilisticTrackingNode::findLaneBoundary(cv::Mat & imgCandidatesHorizontal,
                                                    cv::Mat & imgCandidatesHorizontalTransformed,
                                                    const std::vector<cv::Point2f> & pointsHorizontal)
{
    int printed = 0;
    for (cv::Point2f p : pointsHorizontal) {
        if (p.x >= 0 && p.x < 3000 && p.y >= 0 && p.y < 3500) {
            imgCandidatesHorizontalTransformed.at<uint8_t>((int)p.y, (int)p.x) = 128;
            printed++;
        }
    }
    cv::Size imageSize(imgCandidatesHorizontal.cols, imgCandidatesHorizontal.rows);
    std::vector<cv::Point2f> startPoints;
    if (lane_state == STATE_LANE_RIGHT){
        setUpPossibleWindowsRight(imageSize);
        findStartPointsRight(imgCandidatesHorizontalTransformed, startPoints);
    } else {
        setUpPossibleWindowsLeft(imageSize);
        findStartPointsLeft(imgCandidatesHorizontalTransformed, startPoints);
    }

    std::cout << "Start points found, which is: " << startPoints[0] << "\n"
              << "the converted one is: (" << static_cast<int>(startPoints[0].x) << ", " << static_cast<int>(startPoints[0].y) << ")." << std::endl;
    cv::Rect movingWindow(static_cast<int>(startPoints[0].x), static_cast<int>(startPoints[0].y), transformed_window_h_, transformed_window_w_);
    std::vector<cv::Point2f> pts = slidingWindow(imgCandidatesHorizontalTransformed, movingWindow);
    cv::Mat params = fitLaneBoundary(pts);
    return params;
}

cv::Mat ProbabilisticTrackingNode::findLaneBoundaryForTest(cv::Mat & imgCandidatesHorizontal,
                                                         cv::Mat & imgCandidatesHorizontalTransformed,
                                                         const std::vector<cv::Point2f> & pointsHorizontal,
                                                         const std::string & savePath)
{
    int printed = 0;
    for (cv::Point2f p : pointsHorizontal) {
        if (p.x >= 0 && p.x < 3000 && p.y >= 0 && p.y < 3500) {
            imgCandidatesHorizontalTransformed.at<uint8_t>((int)p.y, (int)p.x) = 128;
            printed++;
        }
    }
    cv::Size imageSize(imgCandidatesHorizontal.cols, imgCandidatesHorizontal.rows);
    std::vector<cv::Point2f> startPoints;
    if (lane_state == STATE_LANE_RIGHT){
        setUpPossibleWindowsRight(imageSize);

        /* //  for testing purpose in terms of initial sliding windows if one of them includes right lane boundary bottom
        cv::Mat imgCandidatesTransformedVisualize = cv::Mat::zeros(cv::Size(3000, 3500), CV_8U);
        cv::cvtColor(imgCandidatesTransformedVisualize, imgCandidatesTransformedVisualize, cv::COLOR_GRAY2BGR);
        std::vector<cv::Point2f> pointsVertical;
        printTransformed(pointsVertical, pointsHorizontal, imgCandidatesTransformedVisualize);
        bool check_3 = cv::imwrite(std::filesystem::path(savePath).parent_path() / "transformed_candidates_with_initial_sw.png", imgCandidatesTransformedVisualize);
        */

        findStartPointsRight(imgCandidatesHorizontalTransformed, startPoints);
    } else {
        setUpPossibleWindowsLeft(imageSize);
        findStartPointsLeft(imgCandidatesHorizontalTransformed, startPoints);
    }

    cv::Rect movingWindow(static_cast<int>(startPoints[0].x), static_cast<int>(startPoints[0].y), transformed_window_h_, transformed_window_w_);
    std::vector<cv::Point2f> pts = slidingWindow(imgCandidatesHorizontalTransformed, movingWindow);
    std::cout << "sliding window is finished." << std::endl;
    // printTransformedRightBoundaryLines(pointsHorizontal, pts, savePath);
    cv::Mat params = fitLaneBoundary(pts); // returned size: 4x1
    return params;
}

std::vector<cv::Point2f> ProbabilisticTrackingNode::slidingWindow(cv::Mat & imgCandidatesHorizontalTransformed, cv::Rect & window)
{
    std::vector<cv::Point2f> points;
    const cv::Size imgSize = imgCandidatesHorizontalTransformed.size();
    bool shouldBreak = false;
    int emptyCounter = 0;

    while (true)
    {
        float currentX = window.x + window.width * 0.5f;
        cv::Mat roi = imgCandidatesHorizontalTransformed(window);
        std::vector<cv::Point2f> locations;
        cv::findNonZero(roi, locations); //Get all non-black pixels. All are white in our case
        float avgX = 0.0f;
        for (int i = 0; i < locations.size(); ++i) { //Calculate average X position
            float x = locations[i].x;
            avgX += window.x + x;
        }
        avgX = locations.empty() ? currentX : avgX / locations.size();
        emptyCounter = locations.empty() ? emptyCounter + 1 : emptyCounter;
        if (emptyCounter == 2) {
            shouldBreak = true;
            if (shouldBreak)
                break;
        }
        cv::Point2f point(avgX, window.y + window.height * 0.5f);
        if (emptyCounter == 0)
            points.push_back(point);
        if (emptyCounter == 1){
            cv::Point2f pointModified(point.x - window.width * 0.5f, window.y + window.height * 0.5f);
            points.push_back(pointModified);
        }
        window.y -= window.height; //Move the window up
        if (window.y < 0){ //For the uppermost position
            window.y = 0;
            shouldBreak = true;
        }
        window.x += (point.x - currentX); //Move x position
        if (window.x < 0) //Make sure the window doesn't overflow, we get an error if we try to get data outside the matrix
            window.x = 0;
        if (window.x + window.width >= imgSize.width)
            window.x = imgSize.width - window.width - 1;
        if (shouldBreak)
            break;
    }
    return points;
}

cv::Mat ProbabilisticTrackingNode::fitLaneBoundary(std::vector<cv::Point2f> & pts)
{
    cv::Mat labels;
    std::vector<cv::Point2f> ctrs;
    cv::kmeans(pts, 4, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, ctrs);
    float xs[16] = {pow(ctrs[0].y, 3), pow(ctrs[0].y, 2), ctrs[0].y, 1,
                    pow(ctrs[1].y, 3), pow(ctrs[1].y, 2), ctrs[1].y, 1,
                    pow(ctrs[2].y, 3), pow(ctrs[2].y, 2), ctrs[2].y, 1,
                    pow(ctrs[3].y, 3), pow(ctrs[3].y, 2), ctrs[3].y, 1};
    cv::Mat data_x = cv::Mat(4, 4, CV_32F, xs);
    data_x = data_x.inv();
    float ys[4] = {ctrs[0].x, ctrs[1].x, ctrs[2].x, ctrs[3].x};
    cv::Mat data_y = cv::Mat(4, 1, CV_32F, ys);
    cv::Mat params = data_x * data_y;
    return params;
}

void ProbabilisticTrackingNode::findStartPointsRight(cv::Mat & imgCandidatesHorizontalTransformed,
                                                std::vector<cv::Point2f> & startPoints)
{
    std::cout << "Finding start points for right lane begins." << std::endl;
    std::vector<cv::Point2f> unfiltered_startpoints;

    for (cv::Rect & r : bottom_windows_) {
        cv::Mat regionOfInterest = imgCandidatesHorizontalTransformed(r);
        std::vector<cv::Point2f> nonBlackPixels;
        cv::findNonZero(regionOfInterest, nonBlackPixels);
        if (!nonBlackPixels.empty()) {
            cv::Point2f newP(r.x, r.y);
            if (unfiltered_startpoints.size() > 0) {
                if (cv::norm(unfiltered_startpoints[unfiltered_startpoints.size() - 1] - newP) < minPointDistance_)
                {
                    continue;
                }
            }
            unfiltered_startpoints.emplace_back(newP);
        }
    }
    if (unfiltered_startpoints.size() == 0)
        std::cout << "no point found from bottom windows" << std::endl;
    else if (unfiltered_startpoints.size() == 1)
        std::cout << "the first unfiltered point is: " << unfiltered_startpoints[0] << std::endl;
    else if (unfiltered_startpoints.size() >= 2){
        std::cout << "the first unfiltered point is: " << unfiltered_startpoints[0] << "\n"
                  << "the second unfiltered point is: " << unfiltered_startpoints[1] << std::endl;
    }
    /**
     * If two points are found at the bottom take the last (i.e. the most right) one. Its highly likely that this is the right lane boundary
     */
    /**
    * Else constraint of the location of possible start point in the right half of transformed image is satisfied
    */
    /**
     * Else look at the right side
     */
    if (unfiltered_startpoints.size() >= 2) {
        startPoints.emplace_back(unfiltered_startpoints[unfiltered_startpoints.size() - 1]);
    } else if (unfiltered_startpoints.size() == 1 && unfiltered_startpoints[0].x > 1500) {
        startPoints.emplace_back(unfiltered_startpoints[0]);
    } else if ((unfiltered_startpoints.size() == 1 && unfiltered_startpoints[0].x <= 1500) ||
                unfiltered_startpoints.size() == 0) {
        for (cv::Rect & r : right_windows_) {
            cv::Mat regionOfInterest = imgCandidatesHorizontalTransformed(r);
            std::vector<cv::Point2f> nonBlackPixels;
            cv::findNonZero(regionOfInterest, nonBlackPixels);
            if (!nonBlackPixels.empty()) {
                startPoints.emplace_back(cv::Point2f(r.x, r.y));
                // lastStartPoints = startPoints;
                break;
            } else {
                // startPoints = lastStartPoints;
            }
        }
    }
    std::cout << "the found start point is: " << startPoints[0] << "\n"
    << "Finding start points is finished." << std::endl;
}

void ProbabilisticTrackingNode::setUpPossibleWindowsRight(const cv::Size & imgSize)
{
    const int input_width_ = imgSize.width;
    const int input_height_ = imgSize.height;
    int current_x = window_w_;
    int current_y = static_cast<int>(start_search_percentile * input_height_);

    while (current_x + window_w_ <= input_width_) {
        std::vector<cv::Point2f> p, q;
        p.emplace_back(cv::Point2f(current_x, current_y));
        cv::perspectiveTransform(p, q, warpmat);
        cv::Rect window = cv::Rect(q[0].x, q[0].y, transformed_window_w_, transformed_window_h_);
        if (window.x < 3000 - transformed_window_w_ && window.x > transformed_window_w_ &&
            window.y < 3500 - transformed_window_h_ && window.y > transformed_window_h_)
        {
            bottom_windows_.emplace_back(window);
        }
        current_x += window_w_;
    }

    current_x = input_width_ - window_w_;
    current_y -= window_h_;
    while (current_x > input_width_ - 5 * window_w_) {
        while (current_y - window_h_ > 0.4 * input_height_) {
            std::vector<cv::Point2f> p, q;
            p.emplace_back(cv::Point2f(current_x, current_y));
            cv::perspectiveTransform(p, q, warpmat);
            cv::Rect window = cv::Rect(q[0].x, q[0].y, transformed_window_w_, transformed_window_h_);
            if (window.x <= 3000 - transformed_window_w_ && window.x > transformed_window_w_ &&
                window.y <= 3500 - transformed_window_h_ && window.y > transformed_window_h_)
            {
                right_windows_.emplace_back(window);
            }
            current_y -= window_h_;
        }
        current_y = static_cast<int>(start_search_percentile * input_height_) - window_h_;
        current_x -= window_w_;
    }
    std::cout << "Setup ends." << std::endl;
}

void ProbabilisticTrackingNode::findStartPointsLeft(cv::Mat & imgCandidatesHorizontalTransformed,
                                                     std::vector<cv::Point2f> & startPoints)
{
    std::cout << "Finding start points for left lane begins." << std::endl;
    std::vector<cv::Point2f> unfiltered_startpoints;

    for (cv::Rect & r : bottom_windows_) {
        cv::Mat regionOfInterest = imgCandidatesHorizontalTransformed(r);
        std::vector<cv::Point2f> nonBlackPixels;
        cv::findNonZero(regionOfInterest, nonBlackPixels);
        if (!nonBlackPixels.empty()) {
            cv::Point2f newP(r.x, r.y);
            if (unfiltered_startpoints.size() > 0) {
                if (cv::norm(unfiltered_startpoints[unfiltered_startpoints.size() - 1] - newP) < minPointDistance_)
                {
                    continue;
                }
            }
            unfiltered_startpoints.emplace_back(newP);
        }
    }
    if (unfiltered_startpoints.size() == 0)
        std::cout << "no point found from bottom windows" << std::endl;
    else if (unfiltered_startpoints.size() == 1)
        std::cout << "the first unfiltered point is: " << unfiltered_startpoints[0] << std::endl;
    else if (unfiltered_startpoints.size() >= 2){
        std::cout << "the first unfiltered point is: " << unfiltered_startpoints[0] << "\n"
                  << "the second unfiltered point is: " << unfiltered_startpoints[1] << std::endl;
    }

    if (unfiltered_startpoints.size() >= 2) {
        startPoints.emplace_back(unfiltered_startpoints[0]);
    } else if (unfiltered_startpoints.size() == 1 && unfiltered_startpoints[0].x < 1500) {
        startPoints.emplace_back(unfiltered_startpoints[0]);
    } else if ((unfiltered_startpoints.size() == 1 && unfiltered_startpoints[0].x >= 1500) ||
               unfiltered_startpoints.size() == 0) {
        for (cv::Rect & r : left_windows_) {
            cv::Mat regionOfInterest = imgCandidatesHorizontalTransformed(r);
            std::vector<cv::Point2f> nonBlackPixels;
            cv::findNonZero(regionOfInterest, nonBlackPixels);
            if (!nonBlackPixels.empty()) {
                startPoints.emplace_back(cv::Point2f(r.x, r.y));
                // lastStartPoints = startPoints;
                break;
            } else {
                // startPoints = lastStartPoints;
            }
        }
    }
    std::cout << "the found start point is: " << startPoints[0] << "\n"
              << "Finding start points is finished." << std::endl;
}

void ProbabilisticTrackingNode::setUpPossibleWindowsLeft(const cv::Size & imgSize)
{
    const int input_width_ = imgSize.width;
    const int input_height_ = imgSize.height;
    int current_x = 0;
    int current_y = static_cast<int>(start_search_percentile * input_height_);

    while (current_x + window_w_ < input_width_ / 2) {
        std::vector<cv::Point2f> p, q;
        p.emplace_back(cv::Point2f(current_x, current_y));
        cv::perspectiveTransform(p, q, warpmat);
        cv::Rect window = cv::Rect(q[0].x, q[0].y, transformed_window_w_, transformed_window_h_);
        if (window.x < 3000 - transformed_window_w_ && window.x >= 0 &&
            window.y < 3500 - transformed_window_h_ && window.y > transformed_window_h_)
        {
            bottom_windows_.emplace_back(window);
        }
        current_x += window_w_;
    }

    current_x = 0;
    current_y -= window_h_;
    while (current_x < 5 * window_w_) {
        while (current_y - window_h_ > 0.4 * input_height_) {
            std::vector<cv::Point2f> p, q;
            p.emplace_back(cv::Point2f(current_x, current_y));
            cv::perspectiveTransform(p, q, warpmat);
            cv::Rect window = cv::Rect(q[0].x, q[0].y, transformed_window_w_, transformed_window_h_);
            if (window.x < 3000 - transformed_window_w_ && window.x >= 0 &&
                window.y <= 3500 - transformed_window_h_ && window.y > transformed_window_h_)
            {
                left_windows_.emplace_back(window);
            }
            current_y -= window_h_;
        }
        current_y = static_cast<int>(start_search_percentile * input_height_) - window_h_;
        current_x += window_w_;
    }
    std::cout << "Setup ends." << std::endl;
}

void ProbabilisticTrackingNode::printTransformedRightBoundaryLines(const std::vector<cv::Point2f> & pointsHorizontal,
                                                                   const std::vector<cv::Point2f> & pts,
                                                                   const std::string & savePath)
{
    cv::Mat in = cv::Mat::zeros(cv::Size(3000, 3500), CV_8U);
    cv::Mat out;
    cv::cvtColor(in, out, cv::COLOR_GRAY2BGR);
    for (cv::Point2f p : pointsHorizontal) {
        if (p.x >= 0 && p.x < 3000 && p.y >= 0 && p.y < 3500) {
            cv::circle(out, p, 5, cv::Scalar(0, 0, 255), 5);
        }
    }
    for (int i = 0; i < pts.size() - 1; ++i) //Draw a line on the processed image
        cv::line(out, pts[i], pts[i + 1], cv::Scalar(0, 255, 0), 4);
    /*
    bool check = cv::imwrite(std::filesystem::path(savePath).parent_path() / "transformed_right_boundary_lines.png", out);
    if (!check) {
        std::cout << "Mission - Saving the transformed right boundary lines, FAILED" << std::endl;
        return;} */
}

void ProbabilisticTrackingNode::removeOutlierFromVecHorizontal(std::vector<cv::Point2f> & pointsHorizontal, const cv::Mat & params)
{
    float threshold = 0.05 * scale;
    for (auto p_it = pointsHorizontal.begin(); p_it != pointsHorizontal.end(); ){
        float xs[4] = {pow((*p_it).y, 3), pow((*p_it).y, 2), (*p_it).y, 1};
        cv::Mat data_x = cv::Mat(4, 1, CV_32F, xs);
        float dist = (*p_it).x - data_x.dot(params);
        if (dist > threshold){
            p_it = pointsHorizontal.erase(p_it);
        } else {
            ++p_it;
        }
    }
}

void ProbabilisticTrackingNode::dynamicHorizontalGradient(const cv::Mat & img, cv::Mat & imgOut)
{
  imgOut = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8U);
  float filterWidth = 3.0f;
  int filterWidthInt = 3;
  int maxFilterWidth = 15;
  float filterStep = (maxFilterWidth - filterWidth) / (3 * img.rows / 4);
  for (int i = img.rows / 4; i < img.rows; ++i) {
    int unnormalizedPrevValue = 0;
    for (int j = filterWidthInt; j < img.cols; ++j) {
      if (j == filterWidthInt) {
        int value = 0;
        for (int k = -filterWidthInt; k < 0; ++k) {
          value -= img.at<uint8_t>(i, j + k);
        }
        for (int k = 1; k <= filterWidthInt; ++k) {
          value += img.at<uint8_t>(i, j + k);
        }
        unnormalizedPrevValue = value;
        value = value / (2 * filterWidthInt);
        value += 127;
        imgOut.at<uint8_t>(i, j) = static_cast<uint8_t>(value);
      } else if (j > filterWidthInt && j < img.cols - filterWidthInt) {
        int value = unnormalizedPrevValue +
          img.at<uint8_t>(i, j + filterWidthInt) -
          img.at<uint8_t>(i, j) +
          img.at<uint8_t>(i, j - filterWidthInt - 1) -
          img.at<uint8_t>(i, j - 1);
        unnormalizedPrevValue = value;
        value /= filterWidthInt * 2;
        value += 127;
        imgOut.at<uint8_t>(i, j) = static_cast<uint8_t>(value);
      }
    }
    filterWidth += filterStep;
    filterWidthInt = static_cast<int>(filterWidth);
  }
}

void ProbabilisticTrackingNode::verticalGradient(const cv::Mat & img, cv::Mat & imgOut) {
  imgOut = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8U);
  float filterWidth = 2.0f;
  int filterWidthInt = 2;
  int maxFilterWidth = 10;
  float filterStep = (maxFilterWidth - filterWidth) / (3 * img.rows / 4);
  int widthDiff = 0;
  for (int j = 0; j < img.cols; ++j) {
    int unnormalizedPrevValue = 0;
    filterWidth = 2.0f;
    filterWidthInt = 2;
    for (int i = img.rows / 4; i < img.rows; ++i) {
      if (i == img.rows / 4) {
        int value = 0;
        for (int k = -filterWidth; k < 0; ++k) {
          value -= img.at<uint8_t>(i + k, j);
        }
        for (int k = 1; k <= filterWidth; ++k) {
          value += img.at<uint8_t>(i + k, j);
        }
        unnormalizedPrevValue = value;
        value = value / (2 * filterWidth);
        value += 127;
        imgOut.at<uint8_t>(i, j) = static_cast<uint8_t>(value);
      } else if (i > img.rows / 4 && i < img.rows - filterWidth) {
        int value = 0;
        if (widthDiff == 0) {
          value = unnormalizedPrevValue +
            img.at<uint8_t>(i + filterWidth, j) -
            img.at<uint8_t>(i, j) +
            img.at<uint8_t>(i - filterWidth - 1, j) -
            img.at<uint8_t>(i - 1, j);
        } else {
          value = unnormalizedPrevValue +
            img.at<uint8_t>(i + filterWidth, j) -
            img.at<uint8_t>(i, j) +
            img.at<uint8_t>(i + filterWidth + 1, j) -
            img.at<uint8_t>(i - 1, j);
        }
        unnormalizedPrevValue = value;
        value /= filterWidth * 2;
        value += 127;
        imgOut.at<uint8_t>(i, j) = static_cast<uint8_t>(value);
      }
      widthDiff = filterWidthInt;
      filterWidth += filterStep;
      filterWidthInt = static_cast<int>(filterWidth);
      widthDiff -= filterWidthInt;
    }
  }
}

void ProbabilisticTrackingNode::findMatchingGradientsHorizontal(const cv::Mat & img, cv::Mat & imgOut, std::vector<cv::Point2f> & measurement)
{
  float distance = 3.0f;
  int distanceInt = 3;
  int maxdistance = 30;
  float distanceStep = (maxdistance - distance) / (3 * img.rows / 4);
  int distanceTolerance = 2;
  int gradientTolerance = 20;
  int threshold = 30;

  if (distanceTolerance > distance) {
    return;
  }

  for (int i = img.rows / 4; i < img.rows; ++i) {
    for (int j = distanceInt + distanceTolerance; j < img.cols; ++j) {
      if (img.at<uint8_t>(i, j) > 127) continue;
      int negativeGradientValue = 127 - img.at<uint8_t>(i, j);
      for (int k = -distanceTolerance; k <= distanceTolerance; ++k) {
        int compareValue = img.at<uint8_t>(i, j - distanceInt + k) - 127;
        if (negativeGradientValue > threshold && compareValue > threshold &&
          abs(compareValue - negativeGradientValue) < gradientTolerance)
        {
          imgOut.at<uint8_t>(i, j - distanceInt / 2) = 128;
          measurement.push_back(cv::Point2f(j, i));
          //j += 30;
          break;
        }
      }
    }
    distance += distanceStep;
    distanceInt = static_cast<int>(distance);
  }
}

void ProbabilisticTrackingNode::findMatchingGradientsVertical(const cv::Mat & img, const cv::Mat & horizontal, cv::Mat & imgOut, std::vector<cv::Point2f> & measurement)
{
  float distance = 3.0f;
  int distanceInt = 3;
  int maxdistance = 30;
  float distanceStep = (maxdistance - distance) / (3 * img.rows / 4);
  int distanceTolerance = 2;
  int gradientTolerance = 20;
  int threshold = 30;
  for (int j = 0; j < img.cols; ++j) {
    distance = 3.0f;
    distanceInt = 3;
    for (int i = img.rows / 4; i < img.rows; ++i) {
      if (img.at<uint8_t>(i, j) > 127) continue;
      int negativeGradientValue = 127 - img.at<uint8_t>(i, j);
      for (int k = -distanceTolerance; k <= distanceTolerance; ++k) {
        int compareValue = img.at<uint8_t>(i - distanceInt + k, j) - 127;
        if (negativeGradientValue > threshold && compareValue > threshold &&
          abs(compareValue - negativeGradientValue) < gradientTolerance)
        {
          if (horizontal.at<uint8_t>(i - distance / 2, j) != 128) {  // already seen in horizontal gradient
            imgOut.at<uint8_t>(i - distance / 2, j) = 128;
            measurement.push_back(cv::Point2f(j, i));
          }
          //j += 30;
          break;
        }
      }
      distance += distanceStep;
      distanceInt = static_cast<int>(distance);
    }
  }
}

void ProbabilisticTrackingNode::classifyAccordingToRulesAdaptive(std::vector<cv::Point2f> & pointsHorizontal,
                                                                 std::vector<cv::Point2f> & pointsVertical,
                                                                 std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings,
                                                                 const cv::Mat & params)
{
    markings.at(TYPE_RIGHT).first = TYPE_RIGHT;
    markings.at(TYPE_LEFT).first = TYPE_LEFT;
    markings.at(TYPE_CENTER_DASHED).first = TYPE_CENTER_DASHED;
    for (cv::Point p : pointsHorizontal) {
        if (p.x < 0 || p.x > 3000 || p.y < 0 || p.y > 3500)
            continue;
        float xs[4] = {pow(p.y, 3), pow(p.y, 2), p.y, 1};
        cv::Mat data_x = cv::Mat(4, 1, CV_32F, xs);
        float dist = p.x - data_x.dot(params);
        // lane width = 0.35 to 0.45 m, so dist to center has to be between 0.175 and 0.225
        // add a little tolerance
        if (lane_state == STATE_LANE_RIGHT) {
            if (abs(dist) < 0.05f * scale) {
                markings.at(TYPE_RIGHT).second.push_back(p);
            } else if (dist > -0.15f * scale && dist < -0.10f * scale) {
                addToSpecials(p, markings, TYPE_NUMBER_ARROW);
            } else if (dist > -0.45f * scale && dist < -0.20f * scale) {
                markings.at(TYPE_CENTER_DASHED).second.push_back(p);
            } else if (dist > -0.85f * scale && dist < -0.55f * scale) {
                markings.at(TYPE_LEFT).second.push_back(p);
            } else if (dist > 0.25f * scale && dist < 0.45f * scale) {
                addToSpecials(p, markings, TYPE_PARKING_PARALLEL);
            }
        } else {
            if (abs(dist) < 0.05f * scale) {
                markings.at(TYPE_LEFT).second.push_back(p);
            } else if (dist > 0.10f * scale && dist < 0.15f * scale) {

            } else if (dist > 0.20f * scale && dist < 0.45f * scale) {
                markings.at(TYPE_CENTER_DASHED).second.push_back(p);
            } else if (dist > 0.55f * scale && dist < 0.85f * scale) {
                markings.at(TYPE_RIGHT).second.push_back(p);
            }
        }
    }
}

void ProbabilisticTrackingNode::printAndClassify(cv::Mat & src, cv::Mat & target)
{
  // define colors for line types
  int thresholdForImportance = 15;
  cv::Vec3b rightBorder(255, 0, 0);
  cv::Vec3b centerLineDashed(255, 127, 0);
  cv::Vec3b green(0, 255, 0);
  int envSize = 5;
  // scan the image bottom up
  int labelCounter = 2;
  int measurementsPerLine = 0;
  std::vector<int> labelAmount;
  std::vector<Eigen::Matrix<float, 2, 1>> labelMean;
  for (int i = src.rows - 10; i >= src.rows / 4; --i) {
    measurementsPerLine = 0;
    for (int j = 0; j < src.cols - envSize; ++j) {
      if (src.at<uint8_t>(i, j) == 1) {
        measurementsPerLine++;
        // scan environment for other labeled pixels
        // its enough to scan to bottom and left directions (rest isnt labeled yet)
        if (labelCounter - 2 > 0) {
          int counters[labelCounter - 2];
          for (int k = 0; k < labelCounter - 2; ++k) {
            counters[k] = 0;
          }
          for (int k = 0; k <= envSize; ++k) {
            for (int l = -envSize; l <= envSize; ++l) {
              uint8_t value = src.at<uint8_t>(i + k, j + l);
              if (value > 1) {
                counters[value - 2]++;
              }
            }
          }

          int maxCounter = 0;
          uint8_t maxLabel = labelCounter + 1;
          for (int k = 0; k < labelCounter - 2; ++k) {
            if (counters[k] > maxCounter) {
              maxCounter = counters[k];
              maxLabel = k + 2;
            }
          }
          src.at<uint8_t>(i, j) = maxLabel;
          if (maxCounter == 0) {
            labelAmount.push_back(0);
            labelMean.push_back(Eigen::Matrix<float, 2, 1>(0, 0));
            labelCounter++;
          }
          labelAmount[maxLabel - 2]++;
          labelMean[maxLabel - 2] += Eigen::Matrix<float, 2, 1>(j, i);
        } else {
          labelAmount.push_back(1);
          labelMean.push_back(Eigen::Matrix<float, 2, 1>(j, i));
          src.at<uint8_t>(i, j) = labelCounter;
          labelCounter++;
        }
      }
    }
  }
  // analyse labels
  // calculate mean for every label
  for (int i = 0; i < labelAmount.size(); ++i) {
    labelMean.at(i)[0] /= labelAmount.at(i);
    labelMean.at(i)[1] /= labelAmount.at(i);
  }
  // calculate covariances
  std::vector<Eigen::Matrix<float, 2, 2>> covariances;
  for (int i = 0; i < labelCounter - 2; ++i) {
    Eigen::Matrix<float, 2, 2> m;
    m << 0, 0, 0, 0;
    covariances.push_back(m);
  }
  for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
      uint8_t value = src.at<uint8_t>(i, j);
      if (value > 1) {
        covariances[value - 2](0, 0) = covariances[value - 2](0, 0) + (j - labelMean[value - 2][0]) * (j - labelMean[value - 2][0]);
        covariances[value - 2](0, 1) = covariances[value - 2](0, 1) + (j - labelMean[value - 2][0]) * (i - labelMean[value - 2][1]);
        covariances[value - 2](1, 1) = covariances[value - 2](1, 1) + (i - labelMean[value - 2][1]) * (i - labelMean[value - 2][1]);
      }
    }
  }
  for (int i = 0; i < covariances.size(); ++i) {
    covariances[i](0, 0) = covariances[i](0, 0) / (labelAmount[i] - 1);
    covariances[i](0, 1) = covariances[i](0, 1) / (labelAmount[i] - 1);
    covariances[i](1, 0) = covariances[i](0, 1);
    covariances[i](1, 1) = covariances[i](1, 1) / (labelAmount[i] - 1);
  }

  std::vector<Eigen::Matrix<std::complex<float>, 2, 2>> eigenvectors;
  std::vector<Eigen::Matrix<std::complex<float>, 2, 1>> eigenvalues;
  Eigen::EigenSolver<Eigen::Matrix<float, 2, 2>> solver;
  for (int i = 0; i < covariances.size(); ++i) {
    if (labelAmount[i] > thresholdForImportance) {
      //std::cout << "amount " << labelAmount[i] << std::endl;
      //std::cout << "mean " << labelMean[i] << std::endl;
      //std::cout << "covariances \n" << covariances.at(i) << std::endl;
      solver.compute(covariances.at(i));
      //std::cout << "eigenvectors \n" << solver.eigenvectors() << std::endl << "eigenvalues " << solver.eigenvalues() << std::endl;
      eigenvectors.push_back(solver.eigenvectors());
      eigenvalues.push_back(solver.eigenvalues());
    } else {
      eigenvectors.push_back(Eigen::Matrix<std::complex<float>, 2, 2>());
      eigenvalues.push_back(Eigen::Matrix<std::complex<float>, 2, 1>());
    }
  }

  // group it!
  std::vector<std::vector<int>> groups;
  for (int i = 0; i < covariances.size(); ++i) {
    bool inserted = false;
    if (labelAmount[i] < thresholdForImportance) continue;
    for (int j = i + 1; j < covariances.size(); ++j) {
      if (labelAmount[j] < thresholdForImportance) continue;
      //std::cout << "val: " << sqrt(eigenvectors[i](0, 0).real() * eigenvectors[j](0, 0).real() + eigenvectors[i](1, 0).real() * eigenvectors[j](1, 0).real()) << std::endl;
      float angle = acos(sqrt(eigenvectors[i](0, 0).real() * eigenvectors[j](0, 0).real() + eigenvectors[i](1, 0).real() * eigenvectors[j](1, 0).real()));
      //std::cout << "acos: " << angle << std::endl;
      if (angle < 10 * M_PI / 360) {
        float angleOfMeansAndCov = acos(sqrt(eigenvectors[i](0, 0).real() * (labelMean[i][0] - labelMean[j][0] + eigenvectors[i](1, 0).real() * (labelMean[i][1] - labelMean[j][1]))));
        if (angle > 10 * M_PI / 360) {
          continue;
        }
        // search if i or j are already in groups
        bool iFound = false;
        int index = 0;
        bool jFound = false;
        for (int k = 0; k < groups.size(); ++k) {
          for (int l = 0; l < groups[k].size(); ++l) {
            if (groups[k][l] == i) {
              iFound = true;
              index = k;
              break;
            } else if (groups[k][l] == j) {
              jFound = true;
              index = k;
              break;
            }
          }
        }
        if (iFound) {
          groups[index].push_back(j);
        } else if (jFound) {
          groups[index].push_back(i);
        } else {
          std::vector<int> vec;
          vec.push_back(i);
          vec.push_back(j);
          groups.push_back(vec);
        }
      }
    }
    if (!inserted) {
      std::vector<int> vec;
      vec.push_back(i);
      groups.push_back(vec);
    }
  }
  std::cout << "groups " << groups.size() << std::endl;

  //std::cout << std::endl;
  for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
      /*switch(src.at<uint8_t>(i, j)) {
        case 1: //target.at<cv::Vec3b>(i, j) = green; break;
                //cv::circle(target, cv::Point(j, i), 2, green, 2); break;
                break;
        case 2: //target.at<cv::Vec3b>(i, j) = centerLineDashed; break;
                cv::circle(target, cv::Point(j, i), 2, centerLineDashed, 2); break;
        case 3: //target.at<cv::Vec3b>(i, j) = rightBorder; break;
                cv::circle(target, cv::Point(j, i), 2, rightBorder, 2); break;
      }*/
      uint8_t value = src.at<uint8_t>(i, j);
      if (labelAmount[value - 2] < thresholdForImportance) continue;
      int groupnr = 0;
      bool broken = false;
      for (int i = 0; i < groups.size(); ++i) {
        for (int j = 0; j < groups[i].size(); ++j) {
          if (groups[i][j] == value - 2) {
            groupnr = i;
            broken = true;
            break;
          }
        }
        if (broken) break;
      }
      switch(groupnr) {
        case 0:
        case 1: break;
        default: {
          if (value % 3 == 0) {
            cv::circle(target, cv::Point(j, i), 2, cv::Vec3b(value % 10 * 25, 0, 0), 2);
          } else if (value % 3 == 1) {
            cv::circle(target, cv::Point(j, i), 2, cv::Vec3b(0, (value % 10) * 25, 0), 2);
          } else {
            cv::circle(target, cv::Point(j, i), 2, cv::Vec3b(0, 0, (value % 10) * 25), 2);
          }
        }
      }
    }
  }

  for (int i = 0; i < labelMean.size(); ++i) {
    if (labelAmount[i] < thresholdForImportance) continue;
    cv::circle(target, cv::Point(labelMean[i][0], labelMean[i][1]), 4, cv::Vec3b(255, 127, 0), 4);
    cv::line(target, cv::Point(labelMean[i][0], labelMean[i][1]), cv::Point(labelMean[i][0] + 10 * eigenvectors[i](0, 0).real(), labelMean[i][1] + 10 * eigenvectors[i](0, 1).real()), cv::Vec3b(0, 127, 255), 2);
    cv::line(target, cv::Point(labelMean[i][0], labelMean[i][1]), cv::Point(labelMean[i][0] + 10 * eigenvectors[i](1, 0).real(), labelMean[i][1] + 10 * eigenvectors[i](1, 1).real()), cv::Vec3b(0, 255, 127), 2);
  }
}

void ProbabilisticTrackingNode::transformPoints(std::vector<cv::Point2f> & src, std::vector<cv::Point2f> & points, const cv::Mat & warpmat)
{
  if (src.size() > 0)
    cv::perspectiveTransform(src, points, warpmat);
}

void ProbabilisticTrackingNode::classifyAccordingToRules(std::vector<cv::Point2f> & pointsHorizontal,
                                                         std::vector<cv::Point2f> & pointsVertical,
                                                         cv::Mat & target,
                                                         std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings)
{
  markings.at(TYPE_RIGHT).first = TYPE_RIGHT;
  markings.at(TYPE_LEFT).first = TYPE_LEFT;
  markings.at(TYPE_CENTER_DASHED).first = TYPE_CENTER_DASHED;
    // check distance to center
    for (cv::Point p : pointsHorizontal) {
      if (p.x < 0 || p.x > 3000 || p.y < 0 || p.y > 3500)
        continue;
      float dist = abs(p.x - 1500);
      // lane width = 0.35 to 0.45 m, so dist to center has to be between 0.175 and 0.225
      // add a little tolerance
      if (dist < 0.3f * scale && dist > 0.15f * scale) { // if in range of center decide according to state
        if (lane_state == STATE_LANE_LEFT) {
          if (p.x < 1500) {
            markings.at(TYPE_LEFT).second.push_back(p);
          } else {
            markings.at(TYPE_CENTER_DASHED).second.push_back(p);
          }
        } else {
          if (p.x < 1500) {
            markings.at(TYPE_CENTER_DASHED).second.push_back(p);
          } else {
            markings.at(TYPE_RIGHT).second.push_back(p);
          }
        }
      } else if (dist < 0.70f * scale && dist > 0.50f * scale) { // markings on other side of road
        if (lane_state == STATE_LANE_LEFT) {
          if (p.x < 1500) {
            addToSpecials(p, markings, TYPE_PARKING_PERPENDICULAR);
          } else {
            markings.at(TYPE_RIGHT).second.push_back(p);
          }
        } else {
          if (p.x < 1500) {
            markings.at(TYPE_LEFT).second.push_back(p);
          } else {
            addToSpecials(p, markings, TYPE_PARKING_PARALLEL);
          }
        }
      } else if (dist < 0.05f * scale){
        addToSpecials(p, markings, TYPE_NUMBER_ARROW);
      }
    }

    for (cv::Point p : pointsVertical) {
      if (p.x < 0 || p.x > 3000 || p.y < 0 || p.y > 3500)
        continue;
      float dist = abs(p.x - 1500);
      if (dist < 0.15f * scale){
        addToSpecials(p, markings, TYPE_HALT_STOP);
      } else {
        addToSpecials(p, markings, TYPE_CROSSING);
      }
    }

  // classify center line
  float min = 3000;
  float max = 0;
  float minY = 3000;
  float maxY = 0;
  for (cv::Point2f p : markings.at(TYPE_CENTER_DASHED).second) { // check if it is to wide for a single marking
    if (p.y > 1500 && p.y < 3000) {
      if (p.x > max) {
        max = p.x;
        maxY = p.y;
      }
      if (p.x < min) {
        min = p.x;
        minY = p.y;
      }
    }
  }
  std::cout << min << ", " << max << ", " << 0.045f * scale << std::endl;
  if (max - min > 0.045f * scale) {
    // count left and right
    float center = min + (max - min) / 2;
    int minCounter = 0;
    int maxCounter = 0;
    for (cv::Point2f p : markings.at(TYPE_CENTER_DASHED).second) {
      if (p.y > 1500) {
        if (p.x > center)
          maxCounter++;
        else
          minCounter++;
      }
    }
    if (abs(maxCounter - minCounter) < 150) {
      markings.at(TYPE_CENTER_DASHED).first = TYPE_CENTER_DOUBLE;
    } else {
      markings.at(TYPE_CENTER_DASHED).first = TYPE_CENTER_COMBINED;
    }
  } // else nothing to do, already dashed type (default)

  // sort crossings and remove perpendicular markings
  int index = -1;
  for (int i = 0; i < markings.size(); ++i) {
    if (markings.at(i).first == TYPE_CROSSING) {
      index = i;
      break;
    }
  }

  if (index > 0) {
    int amount = 5;
    std::vector<cv::Point2f> crossings_vec = markings.at(index).second;
    cv::Mat labels;
    std::vector<cv::Point2f> ctrs;
    cv::kmeans(crossings_vec, amount, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, ctrs);
    int minX[amount] = {3000, 3000, 3000, 3000, 3000};
    int minY[amount] = {3500, 3500, 3500, 3500, 3500};
    int maxX[amount] = {0, 0, 0, 0, 0};
    int maxY[amount] = {0, 0, 0, 0, 0};
    for (int i = 0; i < crossings_vec.size(); ++i) {
      int lbl = labels.at<int>(i);
      if (crossings_vec.at(i).x > maxX[lbl])
        maxX[lbl] = crossings_vec.at(i).x;
      if (crossings_vec.at(i).y > maxY[lbl])
        maxY[lbl] = crossings_vec.at(i).y;
      if (crossings_vec.at(i).x < minX[lbl])
        minX[lbl] = crossings_vec.at(i).x;
      if (crossings_vec.at(i).y < minY[lbl])
        minY[lbl] = crossings_vec.at(i).y;
    }

    markings.at(index).second = std::vector<cv::Point2f>();
    for (int i = 0; i < amount; ++i) {
      if (maxX[i] - minX[i] - (maxY[i] - minY[i]) > 20) { // valid crossing
        for (int j = 0; j < crossings_vec.size(); ++j) {
          if (labels.at<int>(j) == i)
            markings.at(index).second.push_back(crossings_vec[j]);
        }
      }
    }
  }

  // classify halt line
  index = -1;
  for (int i = 0; i < markings.size(); ++i) {
    if (markings.at(i).first == TYPE_HALT_STOP) {
      index = i;
      break;
    }
  }
  if (index > 0) {
    std::vector<cv::Point2f> & halt_vec = markings.at(index).second;
    std::sort(halt_vec.begin(), halt_vec.end(), [](const cv::Point2f & a, const cv::Point2f & b) -> bool {return a.x < b.x;});
    // check for consistancy

    int maxdist = 0;
    for (int i = 1; i < halt_vec.size(); ++i) {
      if (halt_vec[i].x - halt_vec[i-1].x > maxdist)
        maxdist = halt_vec[i].x - halt_vec[i-1].x;
      if (halt_vec[i].x - halt_vec[i-1].x > 0.05f * scale) {
        markings.at(index).first = TYPE_HALT_LOOK;
        break;
      }
    }
  }
}

/*
 * Since there is another method to achieve the goal of sending the essential message, i.e., current angle, this method
 * is now deprecated.
void ProbabilisticTrackingNode::markingsToMessages(const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings, std::vector<libpsaf_msgs::msg::LaneMarkings> & msgVec)
{
  // create functions out of lane borders
  for (int i = 0; i < 3; ++i) {
    //see if markings are fit to fit a function
    float minY = 3500, maxY = 0;
    for (int j = 0; j < markings.at(i).second.size(); ++j) {
      if (markings.at(i).second.at(j).y < minY) {
        minY = markings.at(i).second.at(j).y;
      }
      if (markings.at(i).second.at(j).y > maxY) {
        maxY = markings.at(i).second.at(j).y;
      }
      if (maxY - minY > 2000) {
        cv::Mat labels;
        std::vector<cv::Point2f> ctrs;
        cv::kmeans(markings.at(i).second, 5, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, ctrs);
        float xs[25] = {pow(ctrs[0].y, 4), pow(ctrs[0].y, 3), pow(ctrs[0].y, 2), ctrs[0].y, 1, pow(ctrs[1].y, 4), pow(ctrs[1].y, 3), pow(ctrs[1].y, 2), ctrs[1].y, 1, pow(ctrs[2].y, 4), pow(ctrs[2].y, 3), pow(ctrs[2].y, 2), ctrs[2].y, 1, pow(ctrs[3].y, 4), pow(ctrs[3].y, 3), pow(ctrs[3].y, 2), ctrs[3].y, 1, pow(ctrs[4].y, 4), pow(ctrs[4].y, 3), pow(ctrs[4].y, 2), ctrs[4].y, 1};
        cv::Mat data_x = cv::Mat(5, 5, CV_32F, xs);
        data_x = data_x.inv();
        float ys[5] = {ctrs[0].x, ctrs[1].x, ctrs[2].x, ctrs[3].x, ctrs[4].x};
        cv::Mat data_y = cv::Mat(5, 1, CV_32F, ys);
        cv::Mat params = data_x * data_y;
        libpsaf_msgs::msg::LaneMarkings lm;
        lm.type = markings.at(i).first;
        lm.a = params.at<float>(0); // 4th order
        lm.b = params.at<float>(1); // 3rd order
        lm.c = params.at<float>(2); // 2nd order
        lm.d = params.at<float>(3); // 1st order
        lm.e = params.at<float>(4); // 0th order
        //
      } // else not fit
    }
  }

  for (int i = 3; i < markings.size(); ++i) {
    // get center
    float avgX = 0, avgY = 0;
    for (int j = 0; j < markings.at(i).second.size(); ++j) {
      avgX += markings.at(i).second.at(j).x;
      avgY += markings.at(i).second.at(j).y;
    }
    avgX /= markings.at(i).second.size();
    avgY /= markings.at(i).second.size();
    libpsaf_msgs::msg::LaneMarkings lm;
    lm.position.x = 1500 - avgX;
    lm.position.y = 3500 - avgY;
    lm.type = markings.at(i).first;
    msgVec.push_back(lm);
  }
}
*/

int ProbabilisticTrackingNode::getCourseAngle(const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings,
                                               double & angle, cv::Mat & imgCandidatesTransformed)
{
    float thresholdAngle = 0.349065778; // ~=20°
    bool leftDetected = false;
    bool rightDetected = false;
    cv::Mat paramsLeft, paramsRight;
    if (lane_state == STATE_LANE_RIGHT) {
        rightDetected = getLaneParams(markings.at(0), paramsRight); // (a, b, c ,d)
        leftDetected = getLaneParams(markings.at(2), paramsLeft); // (a, b, c ,d)
    } else {
        rightDetected = getLaneParams(markings.at(2), paramsRight); // (a, b, c ,d)
        leftDetected = getLaneParams(markings.at(1), paramsLeft); // (a, b, c ,d)
    }
    // std::cout << "Lane parameters are found, with rightDetected being " << rightDetected
    //           << " and leftDetected being " << leftDetected << "\n"
    //          << "paramsRight has rows and cols: (" << paramsRight.rows << ", " << paramsRight.cols <<")\n"
    //           << "paramsLeft has rows and cols: (" << paramsLeft.rows << ", " << paramsLeft.cols <<")" << std::endl;
    if (!(!leftDetected && !rightDetected)) {
        cv::Point2f imgBottom = cv::Point2f(3000 / 2, 3000 - 1);  // assumed as the origin of auto coord. sys
        cv::Point2f laneCenter = getLaneCenter(paramsLeft, paramsRight);
        cv::Point2f circleCenter = getCircleCenter(imgBottom, laneCenter);
        float alpha = getAngle(circleCenter, laneCenter);
        angle = courseAngle(circleCenter, imgBottom, alpha / 2);
        if (laneCenter.x > imgBottom.x) {
            angle *= -1;
        }
        std::cout << "last angle: " << lastAngle << ", angle: " << angle << std::endl;
        if (limitAngle) {
            if (abs(lastAngle - angle) < thresholdAngle) {
                lastAngle = angle;
            } else {
                return -1;
            }
        } else {
            lastAngle = angle;
        }
        imgCandidatesTransformed = cv::Mat::zeros(cv::Size(3000, 3500), CV_8U);
        cv::cvtColor(imgCandidatesTransformed, imgCandidatesTransformed, cv::COLOR_GRAY2BGR);
        // cv::circle(imgCandidatesTransformed, circleCenter, abs(circleCenter.x - imgBottom.x), cv::Scalar(127, 255, 127), 3);
        cv::circle(imgCandidatesTransformed, laneCenter, 3, cv::Scalar(255, 165, 0), 5);
        cv::circle(imgCandidatesTransformed, imgBottom, 3, cv::Scalar(255, 165, 0), 5);
        cv::line(imgCandidatesTransformed, imgBottom, laneCenter, cv::Scalar(0, 165, 255), 3);
        cv::circle(imgCandidatesTransformed, circleCenter, 3, cv::Scalar(255, 165, 0), 5);
        return 0;
    }
    return -1;
}

bool ProbabilisticTrackingNode::getLaneParams(const std::pair<int, std::vector<cv::Point2f>> & marking, cv::Mat & params)
{
    bool detected = false;
    float minY = 3500, maxY = 0;
    for (int j = 0; j < marking.second.size(); ++j) {
        if (marking.second.at(j).y < minY) {
            minY = marking.second.at(j).y;
        }
        if (marking.second.at(j).y > maxY) {
            maxY = marking.second.at(j).y;
        }
    }
    if (maxY - minY > 2000) {
        cv::Mat labels;
        std::vector<cv::Point2f> ctrs;
        cv::kmeans(marking.second, 4, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
                   3, cv::KMEANS_PP_CENTERS, ctrs);
        float xs[16] = {pow(ctrs[0].y, 3), pow(ctrs[0].y, 2), ctrs[0].y, 1,
                        pow(ctrs[1].y, 3), pow(ctrs[1].y, 2), ctrs[1].y, 1,
                        pow(ctrs[2].y, 3), pow(ctrs[2].y, 2), ctrs[2].y, 1,
                        pow(ctrs[3].y, 3), pow(ctrs[3].y, 2), ctrs[3].y, 1};
        cv::Mat data_x = cv::Mat(4, 4, CV_32F, xs);
        data_x = data_x.inv();
        float ys[4] = {ctrs[0].x, ctrs[1].x, ctrs[2].x, ctrs[3].x};
        cv::Mat data_y = cv::Mat(4, 1, CV_32F, ys);
        params = data_x * data_y;
        detected = true;
    } else {
        detected = false;
    }
    return detected;
}

cv::Point2f ProbabilisticTrackingNode::getLaneCenter(cv::Mat & paramsLeft, cv::Mat & paramsRight)
{
    float offset = 15;
    float pixelLaneWidth = 0.4 * scale; // width of one lane (Spur)
    float x = 0;
    float y = 1500;
    float ys[4] = {pow(y, 3), pow(y, 2), y, 1};
    cv::Mat y_coeff = cv::Mat(4, 1, CV_32F, ys);
    // std::cout << "y_coeff is: " << y_coeff << ".\n"
    //           << "y_coeff has rows and cols: (" << y_coeff.rows << ", " << y_coeff.cols << ")." << std::endl;
    if (paramsLeft.total() == 4 && paramsRight.total() == 4) {  // both lane sides detected
        float leftValue = y_coeff.dot(paramsLeft); // scalar
        float rightValue = y_coeff.dot(paramsRight); // scalar
        x = (leftValue + rightValue) / 2 - offset;
    } else {
        if (paramsLeft.total() == 4) {
            x = y_coeff.dot(paramsLeft);
            x += pixelLaneWidth / 2 - offset;
        } else {
            x = y_coeff.dot(paramsRight);
            x -= pixelLaneWidth / 2 - offset;
        }
    }
    // std::cout << "the lane center is: (" << x << ", " << y << ")." << std::endl;
    return cv::Point2f(x, y);
}

cv::Point2f ProbabilisticTrackingNode::getCircleCenter(cv::Point2f imgBottom, cv::Point2f laneCenter)
{
    return cv::Point2f(
            ((pow(laneCenter.x, 2) + pow(laneCenter.y, 2) - pow(imgBottom.x, 2) +
              pow(imgBottom.y, 2) - 2 * imgBottom.y * laneCenter.y) /
             (-2 * imgBottom.x + 2 * laneCenter.x)), imgBottom.y);
}

float ProbabilisticTrackingNode::getAngle(cv::Point2f circleCenter, cv::Point2f laneCenter)
{
    return asin((circleCenter.y - laneCenter.y) / cv::norm(laneCenter - circleCenter));
}

float ProbabilisticTrackingNode::courseAngle(cv::Point2f circleCenter, cv::Point2f imgBottom, float alpha)
{
    return atan((1 - cos(alpha)) / sin(alpha));
}

void ProbabilisticTrackingNode::addToSpecials(cv::Point2f p, std::vector<std::pair<int, std::vector<cv::Point2f>>> & specials, int type)
{
  int index = -1;
  for (int i = 0; i < specials.size(); ++i) {
    if (type == specials.at(i).first) {
      index = i;
      break;
    }
  }

  if (index >= 0) {
    specials.at(index).second.push_back(p);
  } else {
    std::vector<cv::Point2f> vec(1);
    vec.at(0) = p;
    specials.push_back(std::pair<int, std::vector<cv::Point2f>>(type, vec));
  }
}

void ProbabilisticTrackingNode::assumeLaneState(
        const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings,
        uint8_t & lane_state)
{
    std::cout << "assuming the lane state begins." << std::endl;
    std::vector<float> delta_ys; //  contain two y_difference separately in order of detected right and left borders
    if (markings.at(0).second.size() == 0 && //  right border
        markings.at(1).second.size() == 0 && //  left border
        markings.at(2).second.size() == 0) { //  center dashed
        lane_state = lane_state;
    } else {
        if (markings.at(0).second.size() == 0 && markings.at(1).second.size() != 0) {
            lane_state = STATE_LANE_RIGHT;
        } else {
            float minY = 3500, maxY = 0;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < markings.at(i).second.size(); ++j) {
                    if (markings.at(i).second.at(j).y < minY) {
                        minY = markings.at(i).second.at(j).y;
                    }
                    if (markings.at(i).second.at(j).y > maxY) {
                        maxY = markings.at(i).second.at(j).y;
                    }
                }
                delta_ys.push_back(maxY - minY);
                minY = 3500;
                maxY = 0;
            }
            // std::cout << "the y difference of right and left borders from top to bottom are separately: ("
            //           << delta_ys.at(0) << ", " << delta_ys.at(1) << ")." << std::endl;
            if (delta_ys.at(0) >= delta_ys.at(1)) {
                lane_state = STATE_LANE_RIGHT;
            } else {
                lane_state = STATE_LANE_LEFT;
            }
        }
    }
}

void ProbabilisticTrackingNode::printCandidates(const cv::Mat & src, cv::Mat & target)
{
  if (src.cols != target.cols || src.rows != target.rows) {
    return;
  }
  cv::Vec3b green(0, 255, 0);
  for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
      if (src.at<uint8_t>(i, j) == 1) {
        target.at<cv::Vec3b>(i, j) = green;
        //cv::circle(target, cv::Point(j, i), 2, green, 2);
      }
    }
  }
}

void ProbabilisticTrackingNode::printParticles(const std::array<Eigen::Vector3f, PARTICLES> & particles, cv::Mat & target)
{
  cv::Vec3b red(255, 0, 0);
  for (int i = 0; i < particles.size(); ++i) {
    //target.at<cv::Vec3b>(particles[i][0], particles[i][1]) = red;
    cv::circle(target, cv::Point(particles[i][1], particles[i][0]), 4, red, 2);
  }
}

void ProbabilisticTrackingNode::printTransformed(const std::vector<cv::Point2f> & pointsVertical, const std::vector<cv::Point2f> & pointsHorizontal, cv::Mat & imgCandidatesTransformed)
{
  int printed = 0;
  cv::Vec3b color = cv::Vec3b(0, 0, 255);
  // for better visualization, it would be better not depict results in one image.
  /*
  for (cv::Point2f p : pointsVertical) {
    if (p.x >= 0 && p.x < 3000 && p.y >= 0 && p.y < 3500) {
      // imgCandidatesTransformed.at<uint8_t>((int)p.y, (int) p.x) = 255;
      cv::circle(imgCandidatesTransformed, p, 5, color, 5);
      printed++;
    }
  }
   */
  for (cv::Point2f p : pointsHorizontal) {
    if (p.x >= 0 && p.x < 3000 && p.y >= 0 && p.y < 3500) {
      // imgCandidatesTransformed.at<uint8_t>((int)p.y, (int) p.x) = 255;
      cv::circle(imgCandidatesTransformed, p, 5, color, 5);
      printed++;
    }
  }
  // add the initial sliding windows for testing purpose
  for (cv::Rect & r : bottom_windows_) {
      cv::rectangle(imgCandidatesTransformed, r, cv::Scalar(0, 255, 0), 4);
  }
  for (cv::Rect & r : right_windows_) {
      cv::rectangle(imgCandidatesTransformed, r, cv::Scalar(0, 255, 0), 4);
  }
}

void ProbabilisticTrackingNode::printMarkings(const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings, cv::Mat & target)
{
  for (std::pair<int, std::vector<cv::Point2f>> marking : markings) {
    for (cv::Point2f p : marking.second) {
      cv::Vec3b color;
      switch (marking.first) {
        case TYPE_RIGHT: color = cv::Vec3b(255, 0, 0); break;
        case TYPE_LEFT: color = cv::Vec3b(0, 255, 0); break;
        case TYPE_CENTER_DASHED: color = cv::Vec3b(0, 0, 255); break;
        case TYPE_CENTER_DOUBLE: color = cv::Vec3b(0, 128, 128); break;
        case TYPE_CENTER_COMBINED: color = cv::Vec3b(0, 128, 255); break;
        case TYPE_PARKING_PERPENDICULAR: color = cv::Vec3b(128, 128, 128); break;
        case TYPE_PARKING_PARALLEL: color = cv::Vec3b(128, 128, 128); break;
        case TYPE_HALT_STOP: color = cv::Vec3b(255, 128, 0); break;
        case TYPE_HALT_LOOK: color = cv::Vec3b(128, 128, 0); break;
        case TYPE_CROSSING: color = cv::Vec3b(0, 255, 240); break;
        case TYPE_NUMBER_ARROW: color = cv::Vec3b(255, 255, 0); break;
        case TYPE_RESTRICTED_AREA: color = cv::Vec3b(255, 255, 0); break;
      }
      if (p.x >= 0 && p.x < 3000 && p.y >= 0 && p.y < 3500)
        //target.at<cv::Vec3b>((int)p.y, (int) p.x) = color;
        cv::circle(target, p, 5, color, 5);
    }
  }
}

void ProbabilisticTrackingNode::printMarkingsForTest(const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings,
                                                     cv::Mat & target, const std::string & savePath, bool imgProc, cv::VideoWriter& video)
{
  for (std::pair<int, std::vector<cv::Point2f>> marking : markings) {
    for (cv::Point2f p : marking.second) {
      cv::Vec3b color;
      switch (marking.first) {
        case TYPE_RIGHT: color = cv::Vec3b(255, 0, 0); break;
        case TYPE_LEFT: color = cv::Vec3b(0, 255, 0); break;
        case TYPE_CENTER_DASHED: color = cv::Vec3b(0, 0, 255); break;
        case TYPE_CENTER_DOUBLE: color = cv::Vec3b(0, 128, 128); break;
        case TYPE_CENTER_COMBINED: color = cv::Vec3b(0, 128, 255); break;
        case TYPE_PARKING_PERPENDICULAR: color = cv::Vec3b(128, 128, 128); break;
        case TYPE_PARKING_PARALLEL: color = cv::Vec3b(128, 128, 128); break;
        case TYPE_HALT_STOP: color = cv::Vec3b(255, 128, 0); break;
        case TYPE_HALT_LOOK: color = cv::Vec3b(128, 128, 0); break;
        case TYPE_CROSSING: color = cv::Vec3b(0, 255, 240); break;
        case TYPE_NUMBER_ARROW: color = cv::Vec3b(255, 255, 0); break;
        case TYPE_RESTRICTED_AREA: color = cv::Vec3b(255, 255, 0); break;
      }
      if (p.x >= 0 && p.x < 3000 && p.y >= 0 && p.y < 3500){
        // target.at<cv::Vec3b>((int)p.y, (int) p.x) = color;
        cv::circle(target, p, 5, color, 5);
      }
    }
  }
  /*
  if(imgProc){
    bool check = cv::imwrite(savePath, target);
    if (!check) {
    std::cout << "Mission - Saving the image, FAILED" << std::endl;
    return;
    }
  }else{
    video.write(target);
    cv::namedWindow("Live", cv::WINDOW_NORMAL);
    cv::resizeWindow("Live", 1200, 1400);
    cv::imshow("Live", target);
    cv::waitKey(1);
  } */

  // cv::imshow("lane markings detection", target);
}

void ProbabilisticTrackingNode::updateTrackingInfo(libpsaf_msgs::msg::TrackingInfo::SharedPtr p)
{
}

void ProbabilisticTrackingNode::onStateChange(int prevState, int newState)
{
}
