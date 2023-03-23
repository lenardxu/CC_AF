#ifndef COMPUTER_VISION__SLIDING_WINDOW_NODE_HPP_
#define COMPUTER_VISION__SLIDING_WINDOW_NODE_HPP_

#define IMG_WIDTH 640
#define IMG_HEIGHT 480

#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"

#include "opencv2/core.hpp"
#include "libpsaf/interface/lane_detection_interface.hpp"

class SlidingWindowNode : public libpsaf::LaneDetectionInterface
{
public:
  SlidingWindowNode()
  : LaneDetectionInterface("sliding_window_node",
      1, std::vector<std::string>(
        {"/color/image_raw"}), "tracking_node/tracking", "state_machine/state",
      "lane_detection_interface/image", "rgb8", "lane_detection_interface/lane_marking",rclcpp::SensorDataQoS()) {}
  explicit SlidingWindowNode(std::string name);
  ~SlidingWindowNode() {}

private:
  /**
   * Publisher for the resulting image of lane detection
   * Topic: /lane_detection/output_image
   */
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr resultImagePublisher;

  /**
   * Publisher for the current movement direction
   * Topic: /lane_detection/current_angle
   */
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr currentAnglePublisher;

  /**
   * Publisher for the target angle
   * Topic: /lane_detection/target_angle
   */
  rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr targetAnglePublisher;

  /**
   * both deprecated
   */
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr leftDistancePublisher;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr rightDistancePublisher;

  /**
   * Publisher for the distance to the closest obstacle
   * Topic: /lane_detection/obstacle_distance
   */
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr obstacleDistancePublisher;

  /**
   * Publisher for the resulting image of obstacle detection
   * Topic: /lane_detection/marked_obstacles
   */
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr markedObstaclesPublisher;

  /**
   * Publisher for the assumption of on which lane the car currently is (can be left or right)
   * Topic: /lane_detection/side
   */
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr sidePublisher;

  /**
   * Subscriber for camera images
   * Topic: /camera/color/image_raw
   */
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cameraImageSubscriber;

  int windowWidth = 80;  /**< width of a window for sliding window approach */
  int windowHeight = 48;  /**< height of a window for sliding window approach */

  float lastDistLeft = 0;  /**< deprecated */
  float lastAngleLeft = 0;  /**< deprecated */
  float lastDistRight = 0;  /**< deprecated */
  float lastAngleRight = 0;  /**< deprecated */
  float lastAngle = 0;  /**< angle of last processed image */
  bool limitAngle = true;  /**< flag if the angle difference between two images should be limited*/
  std::string lastSideAssumption = "right";  /**< assumption of last road side */
  int brightnessThreshold = 180;  /**< threshold for brightness */
  int elementWidth = 20;  /**< width for dilate/erode element */
  int counter = 0;
  int xOutOfY[2] = {2, 3};

public:
  /**
   * \brief Callback function for the image subscriber; automatically called by internal ros functions
   * \param p Pointer to the image received by subscriber
   */
  void colorImageReceivedCallback(const sensor_msgs::msg::Image::SharedPtr p);

  /**
  * \brief Preprocessing step top crop out the hood of the car. Necessary due to the different camera placement 
  *        on the Caraolo Cup car
  * \param src source image
  * \param[out] processedImage resulting image after the croping, rescaled to original size
  */
  void preprocessImage(
     const cv::Mat & src, cv::Mat & processedImages);
  

  /**
   * \brief Complete pipeline to process the image starting with preprocessing like converting to
   *        gray level image and edge detection, followed by lane detection and distance to
   *        potential obstacles
   * \param src source image
   * \param[out] result resulting image of lane detection
   * \param[out] markedObstalce resulting image with marked boundaries of obstacles
   * \param[out] angle angle of movement direction in radians
   * \param[out] side indicator on which side of the road the car is
   * \param[out] distanceObstacle distance to closest obstacle in pixels
   */
  int processImage(
    const cv::Mat & src, cv::Mat & result, cv::Mat & markedObstacles, double & angle,
    std::string & side, double & distanceObstacle);

  /**
   * \brief Detect edges in images based on canny edge detection
   * \param src input image
   * \param[out] dst image with detected edges
   */
  void createEdgeImage(const cv::Mat & src, cv::Mat & dst);

  /**
   * \brief Project image to bird's-eye view
   * \param src input image
   * \param[out] dst projected image
   */
  void projectImage(const cv::Mat & src, cv::Mat & dst);

  /**
   * \brief Runs the sliding window approach on the given image
   * \param src input image
   * \param[out] dst output image with marked windows
   * \param[out] centersLeft vector with the centers of the left lane border; size equals 0 if border could not be detected
   * \param[out] centersRigth vector with the centers of the right lane border; size equals 0 if border could not be detected
   */
  int runSlidingWindows(
    const cv::Mat & src, cv::Mat & dst,
    std::vector<cv::Point2f> & centersLeft, std::vector<cv::Point2f> & centersRight);

  /**
   * \brief Performs one upwards slide starting at the given point and then continues in direction
   *        of the previous slide until the centers leave the image region
   * \param startCenter start point for the first center
   * \param[inout] img image on which the windows should be drawn
   * \param[out] output vector of all detected window centers
   */
  void performSlides(
    const cv::Point2f & startCenter, const cv::Mat & img, std::vector<cv::Point2f> & output,
    const bool Found, const bool AboveFound);
  cv::Point2f centerOfWindowAtPoint(
    const cv::Point2f & windowCenter, const cv::Mat & img, int & amount);

  void averagedistance(
    std::vector<cv::Point2f> & lastpoints, std::vector<cv::Point2f> & currentpoints,
    std::vector<cv::Point2f> & outputpoints, bool Found);

  /**
   * \brief Computes the movement angle for the given situation
   * \param centersLeft centers of the left lane border
   * \param centersRight centers of the right lane border
   * \param[out] angle angle of movement direction
   * \param[inout] img image on which the properties of calculation are drawn
   * \return 0 if successfull computation, -1 if too many information are missing
   */
  int getDistancesAndAngles(
    std::vector<cv::Point2f> & centersLeft, std::vector<cv::Point2f> & centersRight,
    double & angle, cv::Mat & img);

  /**
   * \brief Approximates a polynomial model of degree 3 for the given centers
   * \param[inout] img image on which the model is drawn
   * \param distance deprecated
   * \param angle deprecated
   * \param[out] parameters for the polynom x = a*y^3+b*y^2+c*y+d, where a is at position 3,
   *             b at 2, c at 1 and d at 0
   */
  void processCentersVector(
    std::vector<cv::Point2f> & centers, double & distance, double & angle, cv::Mat & img,
    std::vector<float> & params);

  /**
   * \brief deprectated
   */
  cv::Point2f getWeightedVector(std::vector<cv::Point2f> & points);

  /**
   * \brief deprecated
   */
  void computeModel(std::vector<cv::Point2f> & points, float & a, float & b, float & c);

  /**
   * \brief deprecated
   */
  double getValueAt(float a, float b, float c, float y);

  /**
   * \brief deprecated
   */
  double getDerivativeAt(float a, float b, float y);

  /**
   * \brief draws model x = a*y^3+b*y^2+c*y+d on given image
   * \param a parameter a for polynom
   * \param b parameter b for polynom
   * \param c parameter c for polynom
   * \param d parameter d for polynom
   * \param[inout] image on which the polynom is drawn
   */
  void drawModel(float a, float b, float c, float d, cv::Mat & img);

  /**
   * \brief deprecated
   */
  void drawExponential(float A, float r, cv::Mat & img);

  /**
   * \brief computes the center point of the two given functions at IMG_HEIGHT/2 (=240)
   * \param paramsLeft parameters a, b, c, d of cubic function x = a*y^3+b*y^2+c*y+d where a is at index 3 of vector
   */
  cv::Point2f getLaneCenter(std::vector<float> & paramsLeft, std::vector<float> & paramsRight);

  /**
   * \brief calculates the circle center, so that center.y == imgBottom.y and distance(imgBottom, result) == distance(laneCenter, result)
   */
  cv::Point2f getCircleCenter(cv::Point2f imgBottom, cv::Point2f laneCenter);

  /**
   * \brief computes angular aperture of the circle in reference to the x axis
   * \param circleCenter center of the circle
   * \param laneCenter point at the center of the line
   * \return angle of the sector
   */
  float getAngle(cv::Point2f circleCenter, cv::Point2f laneCenter);

  /**
   * \brief computes the course angle with the help of the point on the circle at angle alpha
   * \param circleCenter center of the circle
   * \param imgBottom position of the bottom center of the image
   * \param alpha angle at which the point for reference should be
   * \return angle of the course
   */
  float courseAngle(cv::Point2f circleCenter, cv::Point2f imgBottom, float alpha);

  /**
   * \brief looks for obstacles in current movement direction and returns the distance to the obstacle
   * \param src source image where to look for the obstacle
   * \param dst destination image where the obstacles are marked
   * \param angle angle of movement direction for the according source image
   * \param obstacleDistance output parameter for the distance to the obstacle (returns max value when no obstacle is detected)
   */
  void obstacleDistance(
    const cv::Mat & src, cv::Mat & dst, double & angle, std::vector<cv::Point2f> & centersLeft,
    std::vector<cv::Point2f> & centersRight, double & obstacleDistance);

  /**
   * \brief calculates the angle to the y axis located at the y coordinate of the first given point
   * \param bottom bottom point of the image where the y axis is located
   * \param referencePoint point of which the angle is compted
   * \return angle in rad
   */
  float getAngleToY(cv::Point & bottom, cv::Point & referencePoint);

  /**
   * \brief assumes the side of the road the car is on
   * \param centersLeft detected centers of left side of lane
   * \param centersRight detected centers of right side of lane
   * \param[out] side side of lane (can either be "left" or "right")
   */
  void assumeSide(
    std::vector<cv::Point2f> & centersLeft, std::vector<cv::Point2f> & centersRight,
    std::string & side);

  /**
   * \brief binary search for the closest window center to the given y value
   * \param y y value where to search
   * \param centers center vector in which the value is searched (descending y values assumed!)
   * \return x value of the closest center
   */
  double getClosestXToYValue(double y, std::vector<cv::Point2f> & centers);

  /**
   * \brief Detects the edges in the given image with a Canny Edge Detector
   * \param img Reference to the source image (RGB)
   * \param sensor the sensor on which the image is received (ID matches with topics vector on initialization by order (first topic = sensor 0))
   */
  void processImage(cv::Mat & img, int sensor) override;

  /**
   * \brief updating tracking information
   * \param p pointer to the new information of the car heading and speed
   * */
  void updateTrackingInfo(libpsaf_msgs::msg::TrackingInfo::SharedPtr p) override;

  /**
   * @brief interface for what should happen, when the state changes
   * @param prevStat state up until now
   * @param newState state from now on
   */
  void onStateChange(int prevState, int newState) override;
};

#endif  // COMPUTER_VISION__SLIDING_WINDOW_NODE_HPP_
