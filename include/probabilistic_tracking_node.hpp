#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/range.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float32.hpp"

#include <opencv2/videoio.hpp>
#include "libpsaf/interface/lane_detection_interface.hpp"
#include "libpsaf_msgs/msg/lane_markings.hpp"
#include "condensation.hpp"
#include "Eigen/Dense"


#define PARTICLES 200

class ProbabilisticTrackingNode : public libpsaf::LaneDetectionInterface
{
public:
  ProbabilisticTrackingNode()
  : LaneDetectionInterface("probabilistic_lane_tracking",
      1, std::vector<std::string>(
        {"color/image_raw"}), "tracking_node/tracking", "state_machine/state",
      "lane_detection_interface/image", "rgb8", "lane_detection_interface/lane_marking",
      rclcpp::SystemDefaultsQoS()) {}

  explicit ProbabilisticTrackingNode(std::string name)
  : ProbabilisticTrackingNode(name, false, false) {}
  explicit ProbabilisticTrackingNode(std::string name, bool publishImageOutput, bool testMode);

  ~ProbabilisticTrackingNode();

  /**
    * \brief The callback method that should be called, when a new state enters the pipeline
    * \param p Shared Pointer to the input state
   */
  void stateReceivedCallback(const std_msgs::msg::Int64::SharedPtr p);

  void processImage(cv::Mat & img, int sensor) override;
  void processImageForTest(cv::Mat & img, const std::string & savePath, bool ImgProcFlag, cv::VideoWriter& video);
  void dynamicHorizontalGradient(const cv::Mat & img, cv::Mat & imgOut);
  void verticalGradient(const cv::Mat & img, cv::Mat & imgOut);
  void findMatchingGradientsHorizontal(const cv::Mat & img, cv::Mat & imgOut, std::vector<cv::Point2f> & measurement);
  void findMatchingGradientsVertical(const cv::Mat & img, const cv::Mat & horizontal, cv::Mat & imgOut, std::vector<cv::Point2f> & measurement);
  void printAndClassify(cv::Mat & src, cv::Mat & target);
  void transformPoints(std::vector<cv::Point2f> & src, std::vector<cv::Point2f> & points, const cv::Mat & warpmat);
  void classifyAccordingToRules(std::vector<cv::Point2f> & pointsHorizontal, std::vector<cv::Point2f> & pointsVertical,
                                cv::Mat & target, std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings);
  void classifyAccordingToRulesAdaptive(std::vector<cv::Point2f> & pointsHorizontal, std::vector<cv::Point2f> & pointsVertical,
                                        std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings,
                                        const cv::Mat & params);
  void markingsToMessages(const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings,
                          std::vector<libpsaf_msgs::msg::LaneMarkings> & msgVec);
  void printCandidates(const cv::Mat & src, cv::Mat & target);
  void printParticles(const std::array<Eigen::Vector3f, PARTICLES> & particles, cv::Mat & target);
  void printTransformed(const std::vector<cv::Point2f> & pointsVertical, const std::vector<cv::Point2f> & pointsHorizontal,
                        cv::Mat & imgCandidatesTransformed);
  void printMarkings(const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings, cv::Mat & target);
  void printMarkingsForTest(const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings, cv::Mat & target,
                            const std::string & savePath, bool imgProc, cv::VideoWriter& video);

  void updateTrackingInfo(libpsaf_msgs::msg::TrackingInfo::SharedPtr p) override;
  void onStateChange(int prevState, int newState) override;

  // Variables
  float scale;  //changed from the private to public due to the use in probabilistic_tracking_test_imshow
  cv::Mat warpmat; //changed from the private to public due to the same reason
  int nr_particle; //changed from the private to public due to the same reason
  double meter_to_pixel_ratio;
  float sqaure_width;
  std::vector<libpsaf_msgs::msg::LaneMarkings> test_lane_markings;

private:
  /**
   * \brief Computes the movement angle for the given situation
   * \param markings lane markings, which are right and left borders, as well as center line
   * \param angle angle of movement direction
   * \param img image on which the properties of calculation are drawn
   * \return 0 if successfull computation, -1 if too many information are missing
   */
  int getCourseAngle(const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings,
                      double & angle,
                      cv::Mat & imgCandidatesTransformed);
  /**
    * \brief Computes the parameters of the fitted line
    * \param marking one individual lane marking
    * \param params parameters of the fitted line
    * \return true if params are found, else false
    */
  bool getLaneParams(const std::pair<int, std::vector<cv::Point2f>> & marking, cv::Mat & params);
  /**
    * \brief computes the center point of the two given functions at half of the effective transformed image
    * \param paramsLeft parameters a, b, c, d of cubic function x = a*y^3+b*y^2+c*y+d where a is at index 3 of vector
    */
  cv::Point2f getLaneCenter(cv::Mat & paramsLeft, cv::Mat & paramsRight);
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
    * \brief Add the special lane markings to normal markings
    * \param p point from edge point collection
    * \param specials special or/and normal lane markings
    * \param type lane markings type, e.g., 0 represents right lane border
    */
  void addToSpecials(cv::Point2f p, std::vector<std::pair<int, std::vector<cv::Point2f>>> & specials, int type);
  /**
   * \brief To find the right lane boundary, which is expressed in fitted curve parameter, here of size: 4x1
   * \param Untransformed image with horizontal edge candidates printed
   * \param Transformed image without any horizontal edge candidates printed
   * \param Transformed horizontal edge candidates
   */
  cv::Mat findLaneBoundary(cv::Mat & imgCandidatesHorizontal,
                           cv::Mat & imgCandidatesHorizontalTransformed,
                           const std::vector<cv::Point2f> & pointsHorizontal);
  cv::Mat findLaneBoundaryForTest(cv::Mat & imgCandidatesHorizontal,
                                  cv::Mat & imgCandidatesHorizontalTransformed,
                                  const std::vector<cv::Point2f> & pointsHorizontal,
                                  const std::string & savePath);
  std::vector<cv::Point2f> slidingWindow(cv::Mat & imgCandidatesHorizontalTransformed, cv::Rect & window);
  void findStartPointsRight(cv::Mat & imgCandidatesHorizontalTransformed, std::vector<cv::Point2f> & startPoints);
  void findStartPointsLeft(cv::Mat & imgCandidatesHorizontalTransformed, std::vector<cv::Point2f> & startPoints);
  void setUpPossibleWindowsRight(const cv::Size & imgSize);
  void setUpPossibleWindowsLeft(const cv::Size & imgSize);
  cv::Mat fitLaneBoundary(std::vector<cv::Point2f> & pts);
  /**
   * \brief To remove the outliers in the filtered transformed horizontal edge points collection
   */
  void removeOutlierFromVecHorizontal(std::vector<cv::Point2f> & pointsHorizontal, const cv::Mat & params);
  /**
   * \brief To print out the found boundary lines which are connected between every two points
   */
  void printTransformedRightBoundaryLines(const std::vector<cv::Point2f> & pointsHorizontal,
                                          const std::vector<cv::Point2f> & pts,
                                          const std::string & savePath);
  void assumeLaneState(const std::vector<std::pair<int, std::vector<cv::Point2f>>> & markings, uint8_t & lane_state);

  bool publishImageOutput = true;
  Condensation<PARTICLES, 3> condense;
  bool lastOneValid = false;
  int64_t state;
  uint8_t lane_state = 0;

  const int window_h_ = 14;  // tunable parameter
  const int window_w_ = 19;  // tunable parameter
  const int transformed_window_h_ = 150;  // tunable parameter
  const int transformed_window_w_ = 90;  // tunable parameter
  const float start_search_percentile = 0.97;  // tunable parameter
  const int minPointDistance_ = 500;  // tunable parameter
  std::vector<cv::Rect> bottom_windows_;
  std::vector<cv::Rect> right_windows_;
  std::vector<cv::Rect> left_windows_;
  std::vector<cv::Point2f> lastStartPoints;

  bool limitAngle = true;  /**< flag if the angle difference between two images should be limited*/
  float lastAngle = 0;  /**< angle of last processed image */
  /**
   * Subscriber for camera images
   * Topic: "/camera/color/image_raw"
   */
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cameraImageSubscriber_;

  /**
   * Subscriber for state
   * Topic: "state_machine/state"
   */
  rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr stateSubscriber_;

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
};
