#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "probabilistic_tracking_node.hpp"

/*!
 *
 * \brief Probabilistic Tracking for Autonomous Driving
 *
 * \param argc An integer argument count of the command line arguments
 * \param argv An argument vector of the command line arguments
 * \return Status of the main program
 */
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
//  InitializeMagick(*argv);
  const rclcpp::QoS QOS{rclcpp::KeepLast(1)};

  auto node = std::make_shared<ProbabilisticTrackingNode>("ProbabilisticTracking", true, false);

  rclcpp::WallRate loopRate(20);
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
    loopRate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
