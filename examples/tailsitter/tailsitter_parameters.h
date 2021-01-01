#include "drake/common/eigen_types.h"

namespace drake {
namespace examples {
namespace tailsitter {

class AeroSurfaceParameters {
  double mass_;
  Vector3<double> dims_;
  Vector3<double> surface_normal_;
  double center_of_pressure_;

 public:
  AeroSurfaceParameters() {}
  AeroSurfaceParameters(const double& _mass, const Vector3<double>& _dims,
                        const Vector3<double>& _surface_normal)
      : mass_(_mass), dims_(_dims), surface_normal_(_surface_normal) {}

  void mass(const double& _mass) { mass_ = _mass; }
  double mass() const { return mass_; }

  void center_of_pressure(const double& _center_of_pressure) {
    center_of_pressure_ = _center_of_pressure;
  }

  double center_of_pressure() const { return center_of_pressure_; }

  void dims(const Vector3<double>& _dims) { dims_ = _dims; }
  Vector3<double> dims() const { return dims_; }

  void surface_normal(const Vector3<double>& _surface_normal) {
    surface_normal_ = _surface_normal;
  }
  Vector3<double> surface_normal() const { return surface_normal_; }

  double area() const {
    return (Vector3<double>::Ones() - surface_normal_).dot(dims_);
  }
};

class PropellerParameters {
  double diameter_;
  double thrust_ratio_, moment_ratio_;

 public:
  PropellerParameters() {}
  PropellerParameters(const double& _diameter, const double& _thrust_ratio,
                      const double& _moment_ratio)
      : diameter_(_diameter),
        thrust_ratio_(_thrust_ratio),
        moment_ratio_(_moment_ratio) {}
  double diameter() const { return diameter_; };
  double thrust_ratio() const { return thrust_ratio_; };
  double moment_ratio() const { return moment_ratio_; };
};

class WorldParameters {
 private:
  double gravity_ = 9.81, atmospheric_density_ = 1.204;

 public:
  WorldParameters(const double _gravity, const double _atmospheric_density)
      : gravity_(_gravity), atmospheric_density_(_atmospheric_density) {}
  double atmospheric_density() const { return atmospheric_density_; }
  double gravity() const { return gravity_; }
};

class TailsitterParameters {
 private:
  double mass_;
  WorldParameters world_;
  AeroSurfaceParameters elevon_, wing_;
  PropellerParameters propeller_;
  double propeller_com_distance_;

 public:
  /**
   * aerodynamic chord is at 1/4th of total lifting area. For tailsitter to
   * be acrobatic COM should coincide with aerodynamic chord. To satisfy this
   * condition, combined mass of props and lifting surfaces must be equal.
   */
  TailsitterParameters() : world_(9.81, 1.204) {
    // to be reset later. initially set to facilitate elevon/wing distribution
    mass_ = 0.081;
    wing_.dims(Vector3<double>(0.335, 0.005, 0.115));
    elevon_.dims(Vector3<double>(wing_.dims()(0) / 2, wing_.dims()(1), 0.044));

    wing_.mass(mass_ / 2 *
               (wing_.dims()(2) / (elevon_.dims()(2) + wing_.dims()(2))));
    wing_.center_of_pressure(3 * wing_.dims()(2) / 4);
    wing_.surface_normal(Vector3<double>::UnitZ());

    elevon_.mass((mass_ / 2 - wing_.mass()) / 2);
    elevon_.center_of_pressure(elevon_.dims()(2) / 2);
    elevon_.surface_normal(Vector3<double>::UnitZ());

    const double prop_thrust_ratio = 1.6 / 2 * mass_ * world_.gravity();
    propeller_ = PropellerParameters(0.135, prop_thrust_ratio,
                                     8.72e-3 * prop_thrust_ratio);
    propeller_com_distance_ = wing_.dims()(0) / 2 - propeller_.diameter() / 2;

    mass_ = 2*elevon_.mass()+wing_.mass();
    assert(propeller_com_distance_ > 0);
  }
  double mass() const { return mass_; }
  WorldParameters world() const { return world_; }
  AeroSurfaceParameters wing() const { return wing_; }
  AeroSurfaceParameters elevon() const { return elevon_; }
  PropellerParameters propeller() const { return propeller_; }
  double propeller_com_distance() const { return propeller_com_distance_; }
};  // namespace tailsitter

}  // namespace tailsitter
}  // namespace examples
}  // namespace drake