#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <limits>

#include "sliding_window_node.hpp"
#include "PolynomialRegression.hpp"

#include "cv_bridge/cv_bridge.h"


std::vector<cv::Point2f> temp_rightpoints;
std::vector<cv::Point2f> temp_leftpoints;

SlidingWindowNode::SlidingWindowNode(std::string name)
: libpsaf::LaneDetectionInterface(name,
    1, std::vector<std::string>(
      {"color/image_raw"}), "tracking_node/tracking", "state_machine/state",
    "lane_detection_interface/image", "mono8", "lane_detection_interface/lane_marking",rclcpp::SensorDataQoS())
{
  const rclcpp::SensorDataQoS QOS{rclcpp::KeepLast(1)};

  /*
  cameraImageSubscriber = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/color/image_raw", QOS,
    std::bind(
      &SlidingWindowNode::colorImageReceivedCallback,
      this, std::placeholders::_1));
  */

  resultImagePublisher = this->create_publisher<sensor_msgs::msg::Image>(
    "/lane_detection/output_image", QOS);
  currentAnglePublisher = this->create_publisher<std_msgs::msg::Float32>(
    "/lane_detection/current_angle", QOS);
  targetAnglePublisher = this->create_publisher<sensor_msgs::msg::Range>(
    "/lane_detection/target_angle", QOS);
  leftDistancePublisher = this->create_publisher<std_msgs::msg::Float64>(
    "/lane_detection/left_distance", QOS);
  rightDistancePublisher = this->create_publisher<std_msgs::msg::Float64>(
    "/lane_detection/right_distance", QOS);

  obstacleDistancePublisher = this->create_publisher<std_msgs::msg::Float64>(
    "/lane_detection/obstacle_distance", QOS);
  markedObstaclesPublisher = this->create_publisher<sensor_msgs::msg::Image>(
    "/lane_detection/marked_obstacles", QOS);

  sidePublisher = this->create_publisher<std_msgs::msg::String>(
    "/lane_detection/side", QOS);

  this->declare_parameter("brightnessThreshold");
  this->declare_parameter("elementWidth");
  // this->declare_parameter("xOutOfY");
  get_parameter("brightnessThreshold", brightnessThreshold);
  get_parameter("elementWidth", elementWidth);
  // xOutOfY[0] = get_parameter("xOutOfY").as_integer_array()[0];
  // xOutOfY[1] = get_parameter("xOutOfY").as_integer_array()[1];
  std::cout << "params: " << brightnessThreshold << ", " << elementWidth << ", " <<
    xOutOfY[0] << "/" << xOutOfY[1] << std::endl;
}


void SlidingWindowNode::colorImageReceivedCallback(const sensor_msgs::msg::Image::SharedPtr p)
{
  auto start = std::chrono::high_resolution_clock::now();


  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(p, sensor_msgs::image_encodings::RGB8);
  } catch (cv_bridge::Exception e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception %s", e.what());
    return;
  }

  cv::Mat srcProcessed;
  cv::Mat result;

  cv::Mat markedObstacles;
  double angle;
  std::string side;
  double distanceObstacle;
  preprocessImage(cv_ptr->image,srcProcessed);

  if (processImage(cv_ptr->image, result, markedObstacles, angle, side, distanceObstacle) != 0) {
    return;
  }

  std_msgs::msg::Float32 cAngleMsg;
  cAngleMsg.data = -angle;
  currentAnglePublisher->publish(cAngleMsg);

  sensor_msgs::msg::Range tAngleMsg;
  tAngleMsg.header.stamp = now();
  tAngleMsg.range = 0.0;
  targetAnglePublisher->publish(tAngleMsg);

  std_msgs::msg::Float64 obstacleDistMsg;
  obstacleDistMsg.data = distanceObstacle;
  if (counter > xOutOfY[0]) {
    obstacleDistancePublisher->publish(obstacleDistMsg);
  } else {
    obstacleDistMsg.data = 100000;
    obstacleDistancePublisher->publish(obstacleDistMsg);
  }

  std_msgs::msg::String sideMsg;
  sideMsg.data = side;
  sidePublisher->publish(sideMsg);

  // tmp
  cv::line(
    result, cv::Point(0, IMG_HEIGHT * 2 / 3), cv::Point(IMG_WIDTH, IMG_HEIGHT * 2 / 3),
    cv::Scalar(255, 0, 0), 3);

  cv_ptr->image = result;
  cv_ptr->encoding = sensor_msgs::image_encodings::RGB8;
  resultImagePublisher->publish(*(cv_ptr->toImageMsg()));
  cv::imshow("Test",result);
  cv::waitKey(32);
  cv_ptr->image = markedObstacles;
  markedObstaclesPublisher->publish(*(cv_ptr->toImageMsg()));
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  // std::cout << "duration image processing " << duration.count() << std::endl;
}


void SlidingWindowNode::preprocessImage(const cv::Mat & src, cv::Mat & processedImage)
{
  cv::Size size = src.size();

  cv::Rect roi;
  roi.x = 0;
  roi.y = 0;
  roi.width = size.width;
  roi.height = (int) ((6.0 * (double)size.height) / 8.0);
  
  cv::Mat cropped = src(roi);
  cv::resize(cropped,processedImage,size,0, 0, CV_INTER_LINEAR);

}

int SlidingWindowNode::processImage(
  const cv::Mat & src, cv::Mat & result, cv::Mat & markedObstacles, double & angle,
  std::string & side, double & distanceObstacle)
{

  cv::Size size = src.size();
  //RCLCPP_INFO(get_logger(), "Width: %i  Height: %i",size.width,size.height);
  //cv::imshow("Test",src);
  //cv::waitKey(32);
  cv::Mat edgeImage;
  createEdgeImage(src, edgeImage);
  //cv::imshow("Edge",edgeImage);

  cv::Mat projected;

  projectImage(edgeImage, projected);
  //cv::imshow("Projected",projected);
  //cv::waitKey(32);
  std::vector<cv::Point2f> leftCenters;
  std::vector<cv::Point2f> rightCenters;

  if (runSlidingWindows(projected, result, leftCenters, rightCenters) != 0) {
    return -1;
  }

  if (getDistancesAndAngles(leftCenters, rightCenters, angle, result) != 0) {
    return -1;
  }
  double neg_angle = -1 * angle;
  obstacleDistance(src, markedObstacles, neg_angle, leftCenters, rightCenters, distanceObstacle);
  assumeSide(leftCenters, rightCenters, side);
  return 0;
}

void SlidingWindowNode::createEdgeImage(const cv::Mat & src, cv::Mat & dst)
{
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // remove through blurring
  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0, 0);

  // create element to extract larger structures
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(blurred, blurred, cv::MORPH_ERODE, element, cv::Point(-1, -1));

  cv::Canny(blurred, dst, 50, 180);
}

void SlidingWindowNode::projectImage(const cv::Mat & src, cv::Mat & dst)
{
  /* double warpData[] = {1.187603992074858, 1.290230562564709, -58.39829914478996,
    -0.09896700993593134, 3.184973913690248, -210.3797150392577,
    -0.0001752846706753409, 0.003882309522171301, 1};  // old 480x640 data */
  /*double warpData[] = {1.145879270445575, 1.661838624165651, -74.06665318934738,
    -0.09735875054602676, 3.123751167296238, -196.0690765361987,
    -0.0001745937118499852, 0.003808570639736984, 1};  // new 480x848 data */

    double warpData[] = {-1.06181818e+00, -2.24484848e+00,  6.37032727e+02,
 -4.44089210e-15, -4.36363636e+00,  8.77090909e+02,
 -8.67361738e-18, -7.12121212e-03,  1.00000000e+00}; //Caraolo Cup car warp data

  cv::Mat warpMatrix = cv::Mat(3, 3, CV_64F, warpData);
  cv::warpPerspective(src, dst, warpMatrix, src.size());
}

int SlidingWindowNode::runSlidingWindows(
  const cv::Mat & src, cv::Mat & dst,
  std::vector<cv::Point2f> & centersLeft, std::vector<cv::Point2f> & centersRight)
{
  // collect all detected edge points in the lower third of the image
  std::vector<cv::Point2f> points;
  for (int y = IMG_HEIGHT * 2 / 3; y < IMG_HEIGHT; ++y) {
    for (int x = 0; x < IMG_WIDTH; ++x) {
      if (src.at<uchar>(y, x) > 200) {
        points.push_back(cv::Point2f(x, y));
      }
    }
  }
  cv::Mat labels, centers;
  // cluster points into 3 groups
  int numClusters = 3;
  if (points.size() < 3) {
    return -1;
  }
  cv::kmeans(
    points, numClusters, labels,
    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
    2, cv::KMEANS_RANDOM_CENTERS, centers);

  // look for centers in left and right side of the image, choose the one closest to image corners
  cv::Point2f leftCenter(IMG_WIDTH / 2, 0);
  bool leftCenterFound = false;
  cv::Point2f rightCenter(IMG_WIDTH / 2, 0);
  bool rightCenterFound = false;
  for (int i = 0; i < numClusters; ++i) {
    cv::Point2f center = centers.at<cv::Point2f>(i);
    if (center.x < IMG_WIDTH / 2 && pow(center.x - 200, 2) + pow(center.y - 480, 2) <
      pow(leftCenter.x - 200, 2) + pow(leftCenter.y - 480, 2))
    {
      leftCenter = center;
      leftCenterFound = true;
    }
    if (center.x > IMG_WIDTH / 2 && pow(center.x - 440, 2) + pow(center.y - 480, 2) <
      pow(rightCenter.x - 440, 2) + pow(rightCenter.y - 480, 2))
    {
      rightCenter = center;
      rightCenterFound = true;
    }
  }

  // temporary draw (remove with guard for output window!)
  cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
  for (int i = 0; i < numClusters; i++) {
    cv::circle(dst, centers.at<cv::Point2f>(i), 3, cv::Scalar(0, 0, 255), 2);
  }
  cv::circle(dst, leftCenter, 5, cv::Scalar(0, 255, 0), 3);
  cv::circle(dst, rightCenter, 5, cv::Scalar(0, 255, 0), 3);
  // end temporary draw
  //cv::imshow("Test", dst);
  //cv::waitKey(32);
  // 0.2-1.5ms on average <1ms
  if (leftCenterFound) {
    performSlides(leftCenter, src, centersLeft, leftCenterFound, true);
    for (size_t i = 0; i < centersLeft.size(); ++i) {
      cv::rectangle(
        dst, centersLeft.at(i) - cv::Point2f(windowWidth / 2, windowHeight / 2),
        centersLeft.at(i) + cv::Point2f(windowWidth / 2, windowHeight / 2),
        cv::Scalar(255, 255, 0), 2);
    }
  }

  if (rightCenterFound) {
    performSlides(rightCenter, src, centersRight, rightCenterFound, true);
    for (size_t i = 0; i < centersRight.size(); ++i) {
      cv::rectangle(
        dst, centersRight.at(i) - cv::Point2f(windowWidth / 2, windowHeight / 2),
        centersRight.at(i) + cv::Point2f(windowWidth / 2, windowHeight / 2),
        cv::Scalar(255, 255, 0), 2);
    }
  }


  if ((!leftCenterFound) && temp_leftpoints.size() > 0) {
    averagedistance(temp_rightpoints, centersRight, temp_leftpoints, leftCenterFound);
    if (temp_leftpoints.size() > 0) {
      cv::Point2f center = temp_leftpoints.at(temp_leftpoints.size() - 1);
      performSlides(center, src, temp_leftpoints, leftCenterFound, true);
      for (size_t i = 0; i < temp_leftpoints.size(); ++i) {
        cv::rectangle(
          dst, temp_leftpoints.at(i) - cv::Point2f(windowWidth / 2, windowHeight / 2),
          temp_leftpoints.at(i) + cv::Point2f(windowWidth / 2, windowHeight / 2),
          cv::Scalar(255, 255, 0), 2);
      }
      centersLeft = temp_leftpoints;
    }
  }

  if ((leftCenterFound && (centersLeft.at(centersLeft.size() - 1).y > IMG_HEIGHT * 2 / 3)) &&
    temp_leftpoints.size() > 0)
  {
    averagedistance(temp_rightpoints, centersRight, temp_leftpoints, leftCenterFound);
    int ii = centersLeft.size();
    cv::Point2f Startcenter = temp_leftpoints.at(temp_leftpoints.size() - 1);
    performSlides(Startcenter, src, centersLeft, leftCenterFound, false);
    for (size_t i = ii - 1; i < centersLeft.size(); ++i) {
      cv::rectangle(
        dst, centersLeft.at(i) - cv::Point2f(windowWidth / 2, windowHeight / 2),
        centersLeft.at(i) + cv::Point2f(windowWidth / 2, windowHeight / 2),
        cv::Scalar(255, 255, 0), 2);
    }
  }

  if ((!rightCenterFound) && temp_rightpoints.size() > 0) {
    averagedistance(temp_leftpoints, centersLeft, temp_rightpoints, rightCenterFound);
    if (temp_rightpoints.size() > 0) {
      cv::Point2f center = temp_rightpoints.at(temp_rightpoints.size() - 1);
      performSlides(center, src, temp_rightpoints, rightCenterFound, true);
      for (size_t i = 0; i < temp_rightpoints.size(); ++i) {
        cv::rectangle(
          dst, temp_rightpoints.at(i) - cv::Point2f(windowWidth / 2, windowHeight / 2),
          temp_rightpoints.at(i) + cv::Point2f(windowWidth / 2, windowHeight / 2),
          cv::Scalar(255, 255, 0), 2);
      }
      centersRight = temp_rightpoints;
    }
  }

  if ((rightCenterFound && (centersRight.at(centersRight.size() - 1).y > IMG_HEIGHT * 2 / 3)) &&
    temp_rightpoints.size() > 0)
  {
    averagedistance(temp_leftpoints, centersLeft, temp_rightpoints, rightCenterFound);
    int ii = centersRight.size();
    cv::Point2f Startcenter = temp_rightpoints.at(temp_rightpoints.size() - 1);
    performSlides(Startcenter, src, centersRight, rightCenterFound, false);
    for (size_t i = ii - 1; i < centersRight.size(); ++i) {
      cv::rectangle(
        dst, centersRight.at(i) - cv::Point2f(windowWidth / 2, windowHeight / 2),
        centersRight.at(i) + cv::Point2f(windowWidth / 2, windowHeight / 2),
        cv::Scalar(255, 255, 0), 2);
    }
  }

  temp_leftpoints = centersLeft;
  temp_rightpoints = centersRight;

  return 0;
}
void SlidingWindowNode::averagedistance(
  std::vector<cv::Point2f> & lastpoints, std::vector<cv::Point2f> & currentpoints,
  std::vector<cv::Point2f> & outputpoints, bool Found)
{
  float mean_x_l = 0;
  float mean_y_l = 0;

  float mean_x_c = 0;
  float mean_y_c = 0;

  std::vector<cv::Point2f> outputs;

  for (size_t i = 0; i < lastpoints.size(); ++i) {
    mean_x_l += lastpoints.at(i).x;
    mean_y_l += lastpoints.at(i).y;
  }

  for (size_t i = 0; i < currentpoints.size(); ++i) {
    mean_x_c += currentpoints.at(i).x;
    mean_y_c += currentpoints.at(i).y;
  }

  float x = mean_x_c / currentpoints.size() - mean_x_l / lastpoints.size();
  float y = mean_y_c / currentpoints.size() - mean_y_l / lastpoints.size();

  for (size_t i = 0; i < outputpoints.size(); ++i) {
    if (outputpoints.at(i).y + y > IMG_HEIGHT * 2 / 3) {
      outputs.push_back(cv::Point2f(outputpoints.at(i).x + x, outputpoints.at(i).y + y));
    }
  }

  if (Found && outputpoints.size() > outputs.size()) {
    cv::Point2f up_begin;
    up_begin.x = outputpoints.at(outputs.size()).x + x;
    up_begin.y = outputpoints.at(outputs.size()).y + y;
    outputs.push_back(up_begin);
  }
  outputpoints = outputs;
}

void SlidingWindowNode::performSlides(
  const cv::Point2f & startCenter, const cv::Mat & img,
  std::vector<cv::Point2f> & output, const bool Found, const bool AboveFound)
{
  cv::Point2f previousSlide(0, -(windowHeight / 2 + 1));
  int prevAmount;
  cv::Point2f center;

  if (Found && AboveFound) {
    prevAmount = 0;
    center = centerOfWindowAtPoint(startCenter, img, prevAmount);
    output.push_back(center);
  } else if (!AboveFound) {
    center = startCenter - previousSlide;
    prevAmount = 1;
  } else {
    center = startCenter;
    prevAmount = 1;
  }

  while (
    center.x >= 0 && center.x < IMG_WIDTH && prevAmount != 0 &&
    center.y >= 0 && center.y < IMG_HEIGHT &&
    (previousSlide.x != 0 || previousSlide.y != 0))
  {
    cv::Point2f newCenter1 = center + previousSlide;
    int amount = 0;
    cv::Point2f newCenter2 = centerOfWindowAtPoint(newCenter1, img, amount);
    // currently not needed
    // cv::Point2f weightedNewCenter = cv::Point2f(
    //         (newCenter1.x * prevAmount + newCenter2.x * amount) / (amount + prevAmount),
    //         (newCenter1.y * prevAmount + newCenter2.y * amount) / (amount + prevAmount));
    if (amount != 0) {
      if (abs(newCenter2.x - center.x) < windowHeight / 4 &&
        abs(newCenter2.y - center.y) < windowHeight / 4)
      {
        previousSlide = 2 * previousSlide;
      } else {
        previousSlide = newCenter2 - center;
      }
      prevAmount = amount;
      center = newCenter2;
      output.push_back(center);
    } else {
      center = center + previousSlide;
    }
  }
}

cv::Point2f SlidingWindowNode::centerOfWindowAtPoint(
  const cv::Point2f & windowCenter, const cv::Mat & img, int & amount)
{
  // collect positions of all edge pixels in window at given position
  cv::Point2f summedPoint(0, 0);
  for (int y = std::max(0, static_cast<int>(windowCenter.y - windowHeight / 2));
    y < std::min(IMG_HEIGHT, static_cast<int>(windowCenter.y + windowHeight / 2)); ++y)
  {
    for (int x = std::max(0, static_cast<int>(windowCenter.x - windowWidth / 2));
      x < std::min(IMG_WIDTH, static_cast<int>(windowCenter.x + windowWidth / 2)); ++x)
    {
      if (img.at<uchar>(y, x) > 200) {
        amount++;
        summedPoint.x += x;
        summedPoint.y += y;
      }
    }
  }

  return cv::Point2f(summedPoint.x / amount, summedPoint.y / amount);
}

int SlidingWindowNode::getDistancesAndAngles(
  std::vector<cv::Point2f> & centersLeft, std::vector<cv::Point2f> & centersRight,
  double & angle, cv::Mat & img)
{
  float thresholdAngle = 0.349065778;  // ~20°
  bool leftDetected = false;
  bool rightDetected = false;
  std::vector<float> paramsLeft, paramsRight;
  if (centersLeft.size() > 4) {
    double a, b;  // not needed for that approach but still in method description for another
    processCentersVector(centersLeft, a, b, img, paramsLeft);
    leftDetected = true;
  }
  if (centersRight.size() > 4) {
    double a, b;
    processCentersVector(centersRight, a, b, img, paramsRight);
    rightDetected = true;
  }
  if (!(!leftDetected && !rightDetected)) {
    cv::Point2f imgBottom = cv::Point2f(IMG_WIDTH / 2, IMG_HEIGHT - 1);
    cv::Point2f laneCenter = getLaneCenter(paramsLeft, paramsRight);
    cv::Point2f circleCenter = getCircleCenter(imgBottom, laneCenter);
    float alpha = getAngle(circleCenter, laneCenter);
    angle = courseAngle(circleCenter, imgBottom, alpha / 2);
    if (laneCenter.x > imgBottom.x) {
      angle *= -1;
    }
    // std::cout << "last angle: " << lastAngle << ", angle: " << angle << std::endl;
    if (limitAngle) {
      if (abs(lastAngle - angle) < thresholdAngle) {
        lastAngle = angle;
      } else {
        return -1;
      }
    } else {
      lastAngle = angle;
    }
    cv::circle(img, circleCenter, abs(circleCenter.x - imgBottom.x), cv::Scalar(127, 255, 127), 3);
    cv::circle(img, laneCenter, 3, cv::Scalar(255, 165, 0), 2);
    cv::circle(img, imgBottom, 3, cv::Scalar(255, 165, 0), 2);
    cv::circle(img, circleCenter, 3, cv::Scalar(255, 165, 0), 2);
    return 0;
  }
  return -1;
}

void SlidingWindowNode::processCentersVector(
  std::vector<cv::Point2f> & centers, double & distance, double & angle, cv::Mat & img,
  std::vector<float> & params)
{
  int polynomialDegree = 3;

  std::vector<float> xs;
  std::vector<float> ys;
  for (size_t i = 0; i < centers.size(); ++i) {
    xs.push_back(centers.at(i).x);
    ys.push_back(centers.at(i).y);
  }

  if (PolynomialRegression<float>().fitIt(ys, xs, polynomialDegree, params)) {
    float a = params.at(3);
    float b = params.at(2);
    float c = params.at(1);
    float d = params.at(0);

    if (a != 0) {
      float y = centers.at(0).y;
      distance = a * pow(y, 3) + b * pow(y, 2) + c * y + d;
      angle = atan(3 * a * pow(y, 2) + b * y + c);
      drawModel(a, b, c, d, img);
    }
  }
}

void SlidingWindowNode::computeModel(
  std::vector<cv::Point2f> & points, float & a, float & b, float & c)
{
  // solve for equation x = a * y^2 + b * y + c for a, b, c
  // use first, last and middle point for calculation
  // --> can be later improved with ransac or least linear squares
  cv::Mat xs(3, 1, CV_32F);
  cv::Mat A(3, 3, CV_32F);
  xs.at<float>(0) = points.at(0).x;
  xs.at<float>(1) = points.at(points.size() / 2).x;
  xs.at<float>(2) = points.at(points.size() - 1).x;
  A.at<float>(0, 0) = pow(points.at(0).y, 2);
  A.at<float>(0, 1) = points.at(0).y;
  A.at<float>(0, 2) = 1;
  A.at<float>(1, 0) = pow(points.at(points.size() / 2).y, 2);
  A.at<float>(1, 1) = points.at(points.size() / 2).y;
  A.at<float>(1, 2) = 1;
  A.at<float>(2, 0) = pow(points.at(points.size() - 1).y, 2);
  A.at<float>(2, 1) = points.at(points.size() - 1).y;
  A.at<float>(2, 2) = 1;
  cv::Mat AT;
  cv::invert(A, AT);
  cv::Mat result = AT * xs;
  a = result.at<float>(0);
  b = result.at<float>(1);
  c = result.at<float>(2);
}

double SlidingWindowNode::getValueAt(float a, float b, float c, float y)
{
  return a * pow(y, 2) + b * y + c;
}

double SlidingWindowNode::getDerivativeAt(float a, float b, float y)
{
  return 2 * a * y + b;
}

void SlidingWindowNode::drawModel(float a, float b, float c, float d, cv::Mat & img)
{
  for (int y = 0; y < IMG_HEIGHT; ++y) {
    int value = a * pow(y, 3) + b * pow(y, 2) + c * y + d;
    if (value >= 0 && value < IMG_WIDTH) {
      img.at<cv::Vec3b>(cv::Point(value, y)) = cv::Vec3b(255, 127, 127);
    }
  }
}

void SlidingWindowNode::drawExponential(float A, float r, cv::Mat & img)
{
  std::cout << "A: " << A << ", r: " << r << std::endl;
  for (int x = 0; x < IMG_WIDTH; ++x) {
    int value = A * pow(r, x);
    if (value >= 0 && value < IMG_WIDTH) {
      img.at<cv::Vec3b>(cv::Point(x, value)) = cv::Vec3b(255, 127, 127);
    }
  }
}

cv::Point2f SlidingWindowNode::getLaneCenter(
  std::vector<float> & paramsLeft, std::vector<float> & paramsRight)
{
  float offset = 50;
  float pixelLaneWidth = 400;
  float x = 0;
  float y = IMG_HEIGHT * 0.4;
  if (paramsLeft.size() > 0 && paramsRight.size() > 0) {  // both lane sides detected
    float a1 = paramsLeft.at(3);
    float b1 = paramsLeft.at(2);
    float c1 = paramsLeft.at(1);
    float d1 = paramsLeft.at(0);

    float leftValue = a1 * pow(y, 3) + b1 * pow(y, 2) + c1 * y + d1;
    float a2 = paramsRight.at(3);
    float b2 = paramsRight.at(2);
    float c2 = paramsRight.at(1);
    float d2 = paramsRight.at(0);

    float rightValue = a2 * pow(y, 3) + b2 * pow(y, 2) + c2 * y + d2;
    if (rightValue < leftValue) {
      if (leftValue > IMG_WIDTH / 2) {
        x = rightValue - pixelLaneWidth / 2 - offset;
      } else {
        x = leftValue + pixelLaneWidth / 2 - offset;
      }
    } else {
      x = (leftValue + rightValue) / 2 - offset;
    }
  } else {
    float a, b, c, d;
    std::vector<float> vecref;
    if (paramsLeft.size() > 0) {
      a = paramsLeft.at(3);
      b = paramsLeft.at(2);
      c = paramsLeft.at(1);
      d = paramsLeft.at(0);
    } else {
      a = paramsRight.at(3);
      b = paramsRight.at(2);
      c = paramsRight.at(1);
      d = paramsRight.at(0);
    }
    x = a * pow(y, 3) + b * pow(y, 2) + c * y + d;
    if (paramsLeft.size() > 0) {
      x += pixelLaneWidth / 2 - offset + 10;
    } else {
      x -= pixelLaneWidth / 2 - offset / 2;
    }
  }
  return cv::Point2f(x, y);
}

cv::Point2f SlidingWindowNode::getCircleCenter(cv::Point2f imgBottom, cv::Point2f laneCenter)
{
  return cv::Point2f(
    ((pow(laneCenter.x, 2) + pow(laneCenter.y, 2) - pow(imgBottom.x, 2) +
    pow(imgBottom.y, 2) - 2 * imgBottom.y * laneCenter.y) /
    (-2 * imgBottom.x + 2 * laneCenter.x)), imgBottom.y);
}

float SlidingWindowNode::getAngle(cv::Point2f circleCenter, cv::Point2f laneCenter)
{
  return asin((circleCenter.y - laneCenter.y) / cv::norm(laneCenter - circleCenter));
}

float SlidingWindowNode::courseAngle(cv::Point2f circleCenter, cv::Point2f imgBottom, float alpha)
{
  // return sin(alpha)*(imgBottom.x-circleCenter.x);
  return atan((1 - cos(alpha)) / sin(alpha));
}

void SlidingWindowNode::obstacleDistance(
  const cv::Mat & src, cv::Mat & dst, double & angle, std::vector<cv::Point2f> & centersLeft,
  std::vector<cv::Point2f> & centersRight, double & obstacleDistance)
{
   /*float warpData[] = {1.145879270445575, 1.661838624165651, -74.06665318934738,
    -0.09735875054602676, 3.123751167296238, -196.0690765361987,
    -0.0001745937118499852, 0.003808570639736984, 1};  // new 480x848 data */

    float warpData[] = {-1.06181818e+00, -2.24484848e+00,  6.37032727e+02,
 -4.44089210e-15, -4.36363636e+00,  8.77090909e+02,
 -8.67361738e-18, -7.12121212e-03,  1.00000000e+00}; //Caraolo Cup car warp data


  cv::Mat warpMatrix = cv::Mat(3, 3, CV_32F, warpData).inv();

  std::vector<cv::Point2f> backtransformedCentersLeft;
  for (size_t i = 0; i < centersLeft.size(); ++i) {
    cv::Point2f currentPoint = centersLeft.at(i);
    cv::Mat_<float> pointX3(3, 1);
    pointX3(0, 0) = currentPoint.x;
    pointX3(1, 0) = currentPoint.y;
    pointX3(2, 0) = 1;
    cv::Mat_<float> transformed = warpMatrix * pointX3;
    backtransformedCentersLeft.push_back(
      cv::Point2f(
        transformed(0, 0) / transformed(2, 0),
        transformed(1, 0) / transformed(2, 0)));
  }
  std::vector<cv::Point2f> backtransformedCentersRight;
  for (size_t i = 0; i < centersRight.size(); ++i) {
    cv::Point2f currentPoint = centersRight.at(i);
    cv::Mat_<float> pointX3(3, 1);
    pointX3(0, 0) = currentPoint.x;
    pointX3(1, 0) = currentPoint.y;
    pointX3(2, 0) = 1;
    cv::Mat_<float> transformed = warpMatrix * pointX3;
    backtransformedCentersRight.push_back(
      cv::Point2f(
        transformed(0, 0) / transformed(2, 0),
        transformed(1, 0) / transformed(2, 0)));
  }

  cv::Mat grayLevel;
  cv::cvtColor(src, grayLevel, CV_RGB2GRAY);

  cv::Mat hsvImage;
  cv::cvtColor(src, hsvImage, cv::COLOR_RGB2HSV);
  cv::Scalar upperOrange(255, 255, 255);
  cv::Scalar lowerOrange(105, 105, 105);
  cv::Mat mask;
  cv::inRange(hsvImage, lowerOrange, upperOrange, mask);
  cv::Mat maskedImage;
  cv::bitwise_and(grayLevel, grayLevel, maskedImage, mask);

  // erode and dilate with a rectangle with large width to remove lane markers
  cv::Mat eroded;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(elementWidth, 1));
  // cv::erode(maskedImage, eroded, element);
  cv::Mat dilated;
  cv::Mat dilateElement =
    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(elementWidth, 9));
  cv::dilate(maskedImage, dilated, dilateElement);

  // apply threshold that only white structures are detected
  //  cv::Mat thresholded;
  //  cv::threshold(dilated, thresholded, brightnessThreshold, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(dilated, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);  dst = src.clone();
  std::vector<std::vector<cv::Point>> contoursInDirection;
  for (auto vec : contours) {
    cv::Point maxX(0, 0);
    cv::Point minX(IMG_WIDTH, 0);
    for (cv::Point p : vec) {
      if (p.x < minX.x) {
        minX = p;
      } else if (p.x > maxX.x) {
        maxX = p;
      }
    }

    double closestXLeftToMin =
      getClosestXToYValue(minX.y, backtransformedCentersLeft);
    double closestXLeftToMax =
      getClosestXToYValue(maxX.y, backtransformedCentersLeft);
    double closestXRightToMin =
      getClosestXToYValue(minX.y, backtransformedCentersRight);
    double closestXRightToMax =
      getClosestXToYValue(maxX.y, backtransformedCentersRight);

    // if not both left of left value or right of right value and
    // obstacle is not close to touching image border
    if (minX.x > 5 && maxX.x < IMG_WIDTH - 5 &&
      !(maxX.x < closestXLeftToMax && minX.x < closestXLeftToMin) &&
      !(maxX.x > closestXRightToMax && minX.x > closestXRightToMin))
    {
      // std::cout << "min: " << minX << ", xleft to min " << closestXLeftToMin <<
      //   ", xright to min " << closestXRightToMin << std::endl;
      // std::cout << "max: " << maxX << ", xleft to max " << closestXLeftToMin <<
      //   ", xright to max " << closestXRightToMin << std::endl;
      contoursInDirection.push_back(vec);
      cv::circle(dst, cv::Point2f(closestXLeftToMin, minX.y), 3, cv::Scalar(255, 125, 0), 2);
      cv::circle(dst, cv::Point2f(closestXRightToMin, minX.y), 3, cv::Scalar(255, 125, 0), 2);
      cv::circle(dst, cv::Point2f(closestXLeftToMax, maxX.y), 3, cv::Scalar(255, 125, 0), 2);
      cv::circle(dst, cv::Point2f(closestXLeftToMax, maxX.y), 3, cv::Scalar(255, 125, 0), 2);
    }
  }

  for (size_t i = 0; i < backtransformedCentersLeft.size(); ++i) {
    cv::rectangle(
      dst, backtransformedCentersLeft.at(i) - cv::Point2f(windowWidth / 2, windowHeight / 2),
      backtransformedCentersLeft.at(i) + cv::Point2f(windowWidth / 2, windowHeight / 2),
      cv::Scalar(255, 255, 0), 2);
  }
  for (size_t i = 0; i < backtransformedCentersRight.size(); ++i) {
    cv::rectangle(
      dst, backtransformedCentersRight.at(i) - cv::Point2f(windowWidth / 2, windowHeight / 2),
      backtransformedCentersRight.at(i) + cv::Point2f(windowWidth / 2, windowHeight / 2),
      cv::Scalar(25, 255, 0), 2);
  }

  for (size_t i = 0; i < contours.size(); ++i) {
    cv::drawContours(dst, contours, i, cv::Scalar(0, 0, 255), 2);
  }
  for (size_t i = 0; i < contoursInDirection.size(); ++i) {
    cv::drawContours(dst, contoursInDirection, i, cv::Scalar(255, 0, 0), 2);
  }

  int maxY = 0;
  for (auto vec : contoursInDirection) {
    for (auto p : vec) {
      if (p.y > maxY) {
        maxY = p.y;
      }
    }
  }
  if (maxY == 0) {
    counter--;
    counter = std::max(0, counter);
    obstacleDistance = std::numeric_limits<double>::max();
  } else {
    counter++;
    counter = std::min(xOutOfY[1], counter);
    obstacleDistance = IMG_HEIGHT - maxY;
  }
}

float SlidingWindowNode::getAngleToY(cv::Point & bottom, cv::Point & referencePoint)
{
  return asin(
    (referencePoint.x - bottom.x) /
    sqrt(pow(bottom.x - referencePoint.x, 2) + pow(bottom.y - referencePoint.y, 2)));
}

void SlidingWindowNode::assumeSide(
  std::vector<cv::Point2f> & centersLeft, std::vector<cv::Point2f> & centersRight,
  std::string & side)
{
  if (centersLeft.size() == 0 && centersRight.size() == 0) {
    side = lastSideAssumption;
  } else {
    int maxDistLeft = 0;
    for (int i = 0; i < static_cast<int>(centersLeft.size()) - 1; ++i) {
      int currentDist = abs(centersLeft.at(i).y - centersLeft.at(i + 1).y);
      if (currentDist > maxDistLeft) {
        maxDistLeft = currentDist;
      }
    }
    int maxDistRight = 0;
    for (int i = 0; i < static_cast<int>(centersRight.size()) - 1; ++i) {
      int currentDist = abs(centersRight.at(i).y - centersRight.at(i + 1).y);
      if (currentDist > maxDistRight) {
        maxDistRight = currentDist;
      }
    }
    if (maxDistLeft > maxDistRight && maxDistLeft > 2 * windowHeight) {
      side = "right";
    } else if (maxDistRight > maxDistLeft && maxDistRight > 2 * windowHeight) {
      side = "left";
    } else {
      side = lastSideAssumption;
    }

    lastSideAssumption = side;
  }
}

double SlidingWindowNode::getClosestXToYValue(double y, std::vector<cv::Point2f> & centers)
{
  int delta = centers.size() / 2;
  int pos = centers.size() / 2;
  while (delta > 0 && pos >= 0 && pos < static_cast<int>(centers.size()) - 1) {
    double upperY = centers.at(pos).y;
    double lowerY = centers.at(pos + 1).y;
    delta = std::max(delta / 2, 1);
    if (y >= lowerY && y <= upperY) {
      if (y - lowerY < upperY - y) {
        return centers.at(pos + 1).x;
      } else {
        return centers.at(pos).x;
      }
    } else if (y < lowerY) {
      pos += delta;
    } else {
      pos -= delta;
    }
  }
  if (pos == -1) {
    return centers.at(0).x;
  } else if (pos == static_cast<int>(centers.size()) - 1) {
    return centers.at(centers.size() - 1).x;
  } else {
    return IMG_WIDTH / 2;
  }
}

void SlidingWindowNode::processImage(cv::Mat & img, int channel)
{
  cv_bridge::CvImage out_msg;
  out_msg.encoding = sensor_msgs::image_encodings::RGB8;
  out_msg.image = img;
  out_msg.image.step = img.cols * 3;

  SlidingWindowNode::colorImageReceivedCallback(out_msg.toImageMsg());
}

void SlidingWindowNode::updateTrackingInfo(libpsaf_msgs::msg::TrackingInfo::SharedPtr p)
{
  // nothing yet
}

void SlidingWindowNode::onStateChange(int prevState, int newState)
{
  // nothing yet
}
