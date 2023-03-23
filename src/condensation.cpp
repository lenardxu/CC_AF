#include "condensation.hpp"

template<size_t SIZE>
void Condensation<size_t amount_particles, size_t dimension_state>::filter(float movement_distance, std::array<float, SIZE> & measurement)
{
  createUnweightedParticleSet();
  predict_movement(movement_distance);
  diffuse();
  combineWithMeasurement<SIZE>(measurement);
  validate();
}