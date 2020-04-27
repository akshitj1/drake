#include "drake/manipulation/util/trajectory_utils.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace manipulation {

using trajectories::PiecewisePolynomial;
using trajectories::PiecewiseQuaternionSlerp;

class PiecewiseCubicTrajectoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    times_ = {0, 2, 3, 4};
    samples_.resize(times_.size(), MatrixX<double>::Zero(2, 1));
    samples_[0] << 0, 1;
    samples_[1] << 2, -3;
    samples_[2] << 1.2, 5;
    samples_[3] << -1, 6;

    test_times_ = {times_.front() - 0.2, times_.front(),
                   (times_.front() + times_.back()) / 2., times_.back(),
                   times_.back() + 0.3};

    pos_ = PiecewisePolynomial<double>::CubicWithContinuousSecondDerivatives(
        times_, samples_);
    dut_ = PiecewiseCubicTrajectory<double>(pos_);
    vel_ = pos_.derivative();
    acc_ = vel_.derivative();
  }

  std::vector<double> times_;
  std::vector<MatrixX<double>> samples_;

  PiecewiseCubicTrajectory<double> dut_;
  std::vector<double> test_times_;

  PiecewisePolynomial<double> pos_;
  PiecewisePolynomial<double> vel_;
  PiecewisePolynomial<double> acc_;
};

// Tests get position matches PiecewisePolynomial.
TEST_F(PiecewiseCubicTrajectoryTest, GetPosition) {
  for (double time : test_times_) {
    EXPECT_TRUE(drake::CompareMatrices(pos_.value(time),
                                       dut_.get_position(time), 1e-12,
                                       drake::MatrixCompareType::absolute));
  }
}

// Tests get velocity matches PiecewisePolynomial when time is in range,
// and zero otherwise.
TEST_F(PiecewiseCubicTrajectoryTest, GetVelocity) {
  for (double time : test_times_) {
    VectorX<double> expected = vel_.value(time);
    if (!pos_.is_time_in_range(time)) expected.setZero();

    EXPECT_TRUE(drake::CompareMatrices(expected, dut_.get_velocity(time), 1e-12,
                                       drake::MatrixCompareType::absolute));
  }
}

// Tests get acceleration matches PiecewisePolynomial when time is in range,
// and zero otherwise.
TEST_F(PiecewiseCubicTrajectoryTest, GetAcceleration) {
  for (double time : test_times_) {
    VectorX<double> expected = acc_.value(time);
    if (!pos_.is_time_in_range(time)) expected.setZero();

    EXPECT_TRUE(drake::CompareMatrices(expected, dut_.get_acceleration(time),
                                       1e-12,
                                       drake::MatrixCompareType::absolute));
  }
}

// Tests get_x_trajectory().
TEST_F(PiecewiseCubicTrajectoryTest, GetTrajectories) {
  EXPECT_TRUE(dut_.get_position_trajectory().isApprox(pos_, 1e-12));
  EXPECT_TRUE(dut_.get_velocity_trajectory().isApprox(vel_, 1e-12));
  EXPECT_TRUE(dut_.get_acceleration_trajectory().isApprox(acc_, 1e-12));
}

// Tests get_x_time().
TEST_F(PiecewiseCubicTrajectoryTest, GetEndTimes) {
  EXPECT_EQ(dut_.get_start_time(), pos_.start_time());
  EXPECT_EQ(dut_.get_end_time(), pos_.end_time());
}

// Tests is_approx().
TEST_F(PiecewiseCubicTrajectoryTest, IsApprox) {
  PiecewiseCubicTrajectory<double> equal = dut_;
  EXPECT_TRUE(dut_.is_approx(equal, 1e-12));

  PiecewiseCubicTrajectory<double> not_equal(
      PiecewisePolynomial<double>::CubicWithContinuousSecondDerivatives(
          times_, samples_, Vector2<double>::Zero(), Vector2<double>::Zero()));

  EXPECT_TRUE(!dut_.is_approx(not_equal, 1e-12));
}

class PiecewiseCartesianTrajectoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<double> times = {1, 2};
    std::vector<AngleAxis<double>> rot_samples(times.size());
    std::vector<MatrixX<double>> pos_samples(times.size(),
                                             MatrixX<double>(3, 1));

    rot_samples[0] = AngleAxis<double>(0.3, Vector3<double>::UnitX());
    rot_samples[1] = AngleAxis<double>(-1, Vector3<double>::UnitY());

    pos_samples[0] << 0.3, 0, -0.5;
    pos_samples[1] << 1, -1, 3;

    std::vector<Isometry3<double>> samples(times.size());
    for (size_t i = 0; i < times.size(); ++i) {
      samples[i].fromPositionOrientationScale(pos_samples[i], rot_samples[i],
                                            Vector3<double>::Ones());
    }

    Vector3<double> vel0(Vector3<double>::Zero());
    Vector3<double> vel1(Vector3<double>::Zero());

    dut_ = PiecewiseCartesianTrajectory<
        double>::MakeCubicLinearWithEndLinearVelocity(times, samples, vel0,
                                                      vel1);

    test_times_ = {times.front() - 0.2, times.front(),
                   (times.front() + times.back()) / 2., times.back(),
                   times.back() + 0.3};

    position_ = PiecewiseCubicTrajectory<double>(
        PiecewisePolynomial<double>::CubicWithContinuousSecondDerivatives(
            times, pos_samples, vel0, vel1));
    orientation_ = PiecewiseQuaternionSlerp<double>(times, rot_samples);
  }

  PiecewiseCartesianTrajectory<double> dut_;
  std::vector<double> test_times_;

  PiecewiseCubicTrajectory<double> position_;
  PiecewiseQuaternionSlerp<double> orientation_;
};

// Tests linear velocity starts and ends at zero.
TEST_F(PiecewiseCartesianTrajectoryTest, TestEndLinearVelocity) {
  double t0 = dut_.get_position_trajectory().get_start_time();
  double t1 = dut_.get_position_trajectory().get_end_time();

  EXPECT_TRUE(drake::CompareMatrices(dut_.get_velocity(t0).tail<3>(),
                                     Vector3<double>::Zero(), 1e-12,
                                     drake::MatrixCompareType::absolute));

  EXPECT_TRUE(drake::CompareMatrices(dut_.get_velocity(t1).tail<3>(),
                                     Vector3<double>::Zero(), 1e-12,
                                     drake::MatrixCompareType::absolute));
}

// Tests pose matches that directly interpolated from PiecewiseQuaternionSlerp
// and PiecewiseCubicTrajectory.
TEST_F(PiecewiseCartesianTrajectoryTest, TestPose) {
  for (double time : test_times_) {
    math::RigidTransform<double> expected(orientation_.orientation(time),
                                          position_.get_position(time));
    EXPECT_TRUE(drake::CompareMatrices(dut_.get_pose(time).matrix(),
                                       expected.GetAsMatrix4(), 1e-12,
                                       drake::MatrixCompareType::absolute));
  }
}

// Tests velocity matches that directly interpolated from
// PiecewiseQuaternionSlerp and PiecewiseCubicTrajectory.
TEST_F(PiecewiseCartesianTrajectoryTest, TestVelocity) {
  for (double time : test_times_) {
    Vector6<double> expected;
    expected.head<3>() = orientation_.angular_velocity(time);
    expected.tail<3>() = position_.get_velocity(time);

    if (!orientation_.is_time_in_range(time)) expected.head<3>().setZero();

    EXPECT_TRUE(drake::CompareMatrices(dut_.get_velocity(time), expected, 1e-12,
                                       drake::MatrixCompareType::absolute));
  }
}

// Tests angular acceleration is always zero because of linear interpolation,
// and linear acceleration matches that interpolated from
// PiecewiseCubicTrajectory.
TEST_F(PiecewiseCartesianTrajectoryTest, TestAccelertaion) {
  for (double time : test_times_) {
    EXPECT_TRUE(drake::CompareMatrices(dut_.get_acceleration(time).head<3>(),
                                       Vector3<double>::Zero(), 1e-12,
                                       drake::MatrixCompareType::absolute));
    EXPECT_TRUE(drake::CompareMatrices(dut_.get_acceleration(time).tail<3>(),
                                       position_.get_acceleration(time), 1e-12,
                                       drake::MatrixCompareType::absolute));
  }
}

// Tests getters.
TEST_F(PiecewiseCartesianTrajectoryTest, TestGetTrajectory) {
  EXPECT_TRUE(dut_.get_position_trajectory().is_approx(position_, 1e-12));
  EXPECT_TRUE(dut_.get_orientation_trajectory().is_approx(orientation_, 1e-12));
}

// Tests different constructors.
TEST_F(PiecewiseCartesianTrajectoryTest, TestConstructor) {
  {
    PiecewiseCartesianTrajectory<double> equal(
        dut_.get_position_trajectory(), dut_.get_orientation_trajectory());
    EXPECT_TRUE(equal.is_approx(dut_, 1e-12));
  }

  {
    PiecewiseCartesianTrajectory<double> equal(
        dut_.get_position_trajectory().get_position_trajectory(),
        dut_.get_orientation_trajectory());
    EXPECT_TRUE(equal.is_approx(dut_, 1e-12));
  }
}

// Tests is_approx().
TEST_F(PiecewiseCartesianTrajectoryTest, TestIsApprox) {
  std::vector<double> times = {1, 2, 3};
  std::vector<AngleAxis<double>> rot_samples(times.size());
  std::vector<MatrixX<double>> pos_samples(times.size(), MatrixX<double>(3, 1));
  pos_samples[0] << -3, 1, 0;
  pos_samples[1] << -2, -1, 5;
  pos_samples[2] << -2, -1, 5;

  rot_samples[0] = AngleAxis<double>(0.3, Vector3<double>::UnitX());
  rot_samples[1] = AngleAxis<double>(-1, Vector3<double>::UnitY());
  rot_samples[2] = AngleAxis<double>(-0.44, Vector3<double>::UnitZ());

  PiecewiseCubicTrajectory<double> new_pos_traj(
      PiecewisePolynomial<double>::CubicWithContinuousSecondDerivatives(
          times, pos_samples));
  PiecewiseQuaternionSlerp<double> new_rot_traj(times, rot_samples);

  {
    PiecewiseCartesianTrajectory<double> diff_position(new_pos_traj,
                                                       orientation_);
    EXPECT_TRUE(!diff_position.is_approx(dut_, 1e-12));
  }

  {
    PiecewiseCartesianTrajectory<double> diff_orientation(position_,
                                                          new_rot_traj);
    EXPECT_TRUE(!diff_orientation.is_approx(dut_, 1e-12));
  }
}

}  // namespace manipulation
}  // namespace drake
