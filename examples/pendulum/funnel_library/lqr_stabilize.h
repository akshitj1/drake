#include "Eigen/Core"

std::unique_ptr<drake::systems::controllers::TimeVaryingLQR>
StabilizingLQRController(const Glider<double>* glider, const PPoly& x_des,
                         const PPoly& u_des) {
  const VectorX<double> kXf_err_max{
      (VectorX<double>(7) << 0.05, 0.05, 3, 3, 1, 1, 3).finished()};
  const MatrixX<double> Qf{
      kXf_err_max.array().square().inverse().matrix().asDiagonal()};
  const MatrixX<double> Q{
      (VectorX<double>(7) << 10, 10, 10, 1, 1, 1, 1).finished().asDiagonal()};
  const MatrixX<double> R{(MatrixX<double>(1, 1) << 0.1).finished()};

  return std::make_unique<TimeVaryingLQR>(*glider, x_des, u_des, Q, R, Qf);
}

