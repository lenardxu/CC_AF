#ifndef COMPUTER_VISION__CONDENSATION_HPP_
#define COMPUTER_VISION__CONDENSATION_HPP_

#include <array>
#include "Eigen/Dense"
#include <chrono>
#include <random>

/**
 * Implementation according to paper "Probabilistic Lane Tracking in Difficult Road Scenarios Using Stereovision" by Danescu et al.
 * in IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS, VOL. 10, NO. 2, JUNE 2009
 */
template<size_t amount_particles, size_t dimension_state>
class Condensation
{
public:
  Condensation() {
    for (int i = 0; i < particles.size(); ++i) {
      particles[i] = Eigen::Matrix<float, 3, 1>(240 + i, 424, 0);
      weights[i] = 1 * 1.0f / amount_particles;
      accumulated_weights[i] = (i + 1) * 1.0f / amount_particles;
    }
    /*for (int i = 0; i < particles.size() / 4; ++i) {
      particles[i] = Eigen::Matrix<float, 3, 1>(200, 240, 0);
      weights[i] = 1 * 1.0f / amount_particles;
      accumulated_weights[i] = (i + 1) * 1.0f / amount_particles;
    }
    for (int i = particles.size() / 4; i < particles.size() / 2; ++i) {
      particles[i] = Eigen::Matrix<float, 3, 1>(600, 240, 0);
      weights[i] = 1 * 1.0f / amount_particles;
      accumulated_weights[i] = (i + 1) * 1.0f / amount_particles;
    }
    for (int i = particles.size() / 2; i < 3 * particles.size() / 4; ++i) {
      particles[i] = Eigen::Matrix<float, 3, 1>(200, 120, 0);
      weights[i] = 1 * 1.0f / amount_particles;
      accumulated_weights[i] = (i + 1) * 1.0f / amount_particles;
    }
    for (int i = 3 * particles.size() / 4; i < particles.size(); ++i) {
      particles[i] = Eigen::Matrix<float, 3, 1>(600, 120, 0);
      weights[i] = 1 * 1.0f / amount_particles;
      accumulated_weights[i] = (i + 1) * 1.0f / amount_particles;
    }*/
    /*for (int i = 0; i < particles.size(); ++i) {
      weights[i] = 1 * 1.0f / amount_particles;
      accumulated_weights[i] = (i + 1) * 1.0f / amount_particles;
    }*/

    //particles[0] = Eigen::Matrix<float, 3, 1>(424, 240, 0);
    //particles[1] = Eigen::Matrix<float, 3, 1>(300, 100, 0);
    //particles[2] = Eigen::Matrix<float, 3, 1>(300, 300, 0);
    //particles[3] = Eigen::Matrix<float, 3, 1>(100, 300, 0);
    //particles[4] = Eigen::Matrix<float, 3, 1>(400, 100, 0);

    srand((unsigned int)time(NULL));

    printParticles();
    for (int i = 0; i < particles.size(); ++i) {
      std::cout << accumulated_weights[i] << ", ";
    }
    std::cout << std::endl;
    distribution = std::normal_distribution<float>(0.0, 2.0);
  }

  /**
   * @param curve_radius positive: curve to the left, negative: curve to the right
   */
  void filter(float movement_distance, float curve_radius, std::vector<Eigen::Matrix<int, 2, 1>> & measurement)
  {
    // auto start_time = std::chrono::high_resolution_clock::now();
    createUnweightedParticleSet();
    // auto unweighted_time = std::chrono::high_resolution_clock::now();
    predict_movement(movement_distance, curve_radius);
    //std::cout << "movement predicted" << std::endl;
    //printParticles();
    // auto predicted_time = std::chrono::high_resolution_clock::now();
    combineWithMeasurement(measurement);
    // auto combined_time = std::chrono::high_resolution_clock::now();
    normalizeWeights();
    // auto normalized_time = std::chrono::high_resolution_clock::now();
    validate();

    //printParticles();

    // auto duration_weight = std::chrono::duration_cast<std::chrono::milliseconds>(unweighted_time - start_time);
    // auto duration_predict = std::chrono::duration_cast<std::chrono::milliseconds>(predicted_time - unweighted_time);
    // auto duration_combine = std::chrono::duration_cast<std::chrono::milliseconds>(combined_time - predicted_time);
    // auto duration_normalize = std::chrono::duration_cast<std::chrono::milliseconds>(normalized_time - combined_time);
    // std::cout << "weight: " << duration_weight.count() << ", predict: " << duration_predict.count() << ", combine: " << duration_combine.count() << ", normalize: " << duration_normalize.count() << std::endl;
  }
  
  void createUnweightedParticleSet()
  {
    std::array<Eigen::Matrix<float, dimension_state, 1>, amount_particles> working_set;

    int init_particles = amount_particles * 0.1;
    // draw from old particle set
    for (int i = 0; i < amount_particles - init_particles; ++i) {
      float random_number = (float) rand() / RAND_MAX;
      //std::cout << "picked: " << random_number << std::endl;
      for (int j = 0; j < accumulated_weights.size(); ++j) {
        if (random_number <= accumulated_weights[j]) {
          working_set[i] = particles[j];
          break;
        }
      }
    }

    // draw from intial (random) particle set
    for (int i = amount_particles - init_particles; i < amount_particles; ++i) {
      float coordY = rand() * 1.0f / RAND_MAX * 848;
      float coordX = rand() * 1.0f / RAND_MAX * 360 + 120;
      //std::cout << "rand x: " << coordX << ", rand y: " << coordY << std::endl;
      working_set[i] = Eigen::Matrix<float, dimension_state, 1>(coordX, coordY, 0);
    }

    // copy
    for (int i = 0; i < working_set.size(); ++i) {
      particles[i] = working_set[i];
    }
  }

  void predict_movement(float distance, float curve_radius)
  {
    float angle = distance * 360 / (2 * curve_radius * M_PI);
    if (curve_radius == 0) {
      angle = 0;
    }
    float x_diff = sin(angle) * abs(curve_radius);
    float y_diff = curve_radius - cos(angle) * abs(curve_radius);
    for (int i = 0; i < particles.size(); ++i) {
      float noise = distribution(generator);
      //std::cout << "noise x " << noise << " ";
      particles[i][0] = particles[i][0] - x_diff + noise;
      noise = distribution(generator);
      //std::cout << "noise y " << noise << std::endl;
      if (curve_radius > 0) {
        particles[i][1] = particles[i][1] - y_diff + noise;
      } else {
        particles[i][1] = particles[i][1] + y_diff + noise;
      }
    }
    /*float distance_squared_half = distance * distance / 2;
    float noise = 0;
    for(auto p : particles) {
      p(3, 0) = p(1, 0) * -distance_squared_half + p(3, 0) + p(6, 0) * distance + distance_squared_half * curvature + noise;
      p(6, 0) = p(1, 0) * -distance + p(6, 0) + distance * curvature + noise;
    }*/
  }

  void combineWithMeasurement(std::vector<Eigen::Matrix<int, 2, 1>> & measurement)
  {
    float sigma = 5.0;
    //float tolerance = 1.2;
    float sigma_sq_x2 = sigma * sigma * 2;
    float factor = 1 / (sigma * sqrt(2 * M_PI));
    for (int i = 0; i < amount_particles; ++i) {
      weights[i] = 0;
      for (auto m : measurement) {
        //if (abs(m(1, 0) - particles[i](1, 0)) > tolerance || m(0, 0) - particles[i](0, 0) > tolerance) continue;
        weights[i] += factor * exp(- ((m(0, 0) - particles[i](0, 0)) * (m(0, 0) - particles[i](0, 0)) + (m(1, 0) - particles[i](1, 0)) * (m(1, 0) - particles[i](1, 0))) / sigma_sq_x2);
      }
      //std::cout << "weight " << i << ": " << weights[i] << std::endl;
    }
  }

  void normalizeWeights()
  {
    float sum_weights = 0;
    for (float w : weights) {
      sum_weights += w;
    }
    //std::cout << "sum weights: " << sum_weights << std::endl;
    for (int i = 0; i < weights.size(); ++i) {
      weights[i] /= sum_weights;
      if (i == 0) {
        accumulated_weights[i] = weights[i];
      } else {
        accumulated_weights[i] = accumulated_weights[i - 1] + weights[i];
      }
    }
  }

  void validate()
  {

  }

  std::array<Eigen::Matrix<float, dimension_state, 1>, amount_particles> & getParticles() {
    return particles;
  }

  float getWeight(int index) {
    return weights[index];
  }

  void printParticles() {
    for (int i = 0; i < particles.size(); ++i) {
      std::cout << "particle " << i << ", position: " << particles[i][0] << ", " << particles[i][1] << ", label: " << particles[i][2] << ", weight: " << weights[i] << std::endl;
    }
  }

private:
  std::array<Eigen::Matrix<float, dimension_state, 1>, amount_particles> particles;
  std::array<float, amount_particles> weights;
  std::array<float, amount_particles> accumulated_weights;

  std::default_random_engine generator;
  std::normal_distribution<float> distribution;
};

#endif  // COMPUTER_VISION__CONDENSATION

