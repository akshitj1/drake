#pragma once

#include <memory>
#include <string>

#include "drake/examples/tailsitter3d/external_spatial_force_multiplexer.h"
#include "drake/examples/tailsitter3d/flat_wing.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/propeller.h"

namespace drake {
namespace examples {
namespace tailsitter3d {
using drake::math::RigidTransformd;
using geometry::Box;
using geometry::Cylinder;
using geometry::SceneGraph;
using multibody::ExternalSpatialForceMultiplexer;
using multibody::FlatWing;
using multibody::FlatWingInfo;
using multibody::MultibodyPlant;
using multibody::Propeller;
using multibody::PropellerInfo;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::UnitInertia;

class TailsitterParameters {
 private:
  const double gravity_, atmospheric_density_;
  const double wing_lx_, wing_ly_, wing_lz_;
  const double elevon_lx_, elevon_ly_, elevon_lz_;
  const double wing_aerodynamic_center_;
  const double tailsitter_mass_, wing_mass_, elevon_mass_, propeller_mass_;

  const double propeller_diameter_, propeller_com_distance_,
      propeller_thrust_ratio_, propeller_moment_ratio_;

 public:
  // DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(TailsitterParameters)
  /**
   * aerodynamic chord is at 1/4th of total lifting area. For tailsitter to
   * be acrobatic COM should coincide with aerodynamic chord. To satisfy this
   * condition, combined mass of props and lifting surfaces must be equal.
   */
  TailsitterParameters()
      : gravity_(9.81),
        atmospheric_density_(1.204),
        wing_lz_(0.115),
        wing_lx_(0.335),
        wing_ly_(0.005),
        elevon_lz_(0.044),
        elevon_lx_(wing_lx_ / 2),
        elevon_ly_(wing_ly_),
        tailsitter_mass_(0.081),
        propeller_mass_(tailsitter_mass_ / 4),
        wing_mass_(tailsitter_mass_ / 2 * (wing_lz_ / (elevon_lz_ + wing_lz_))),
        elevon_mass_((tailsitter_mass_ / 2 - wing_mass_) / 2),
        wing_aerodynamic_center_(3 * wing_lz_ / 4),
        propeller_diameter_(0.135),
        propeller_com_distance_(wing_lx_ / 2 - propeller_diameter_ / 2),
        propeller_thrust_ratio_(1.6 / 2 * tailsitter_mass_ * gravity_),
        propeller_moment_ratio_(8.72e-3 * propeller_thrust_ratio_) {
    assert(propeller_com_distance_ > 0);
  }

  double wing_lx() const { return wing_lx_; }
  double wing_ly() const { return wing_ly_; }
  double wing_lz() const { return wing_lz_; }
  double wing_mass() const { return wing_mass_; }
  double tailsitter_mass() const { return tailsitter_mass_; }
  double propeller_com_distance() const { return propeller_com_distance_; }
  double propeller_diameter() const { return propeller_diameter_; }
  double propeller_thrust_ratio() const { return propeller_thrust_ratio_; }
  double propeller_moment_ratio() const { return propeller_moment_ratio_; }
  double wing_aerodynamic_center() const { return wing_aerodynamic_center_; }
  double wing_area() const { return wing_lx_ * wing_lz_; }
  double atmospheric_density() const { return atmospheric_density_; }
  double gravity() const { return gravity_; }
};

class TailsitterPlantBuilder {
 private:
  const TailsitterParameters params;

 public:
  TailsitterPlantBuilder() : params() {}

  TailsitterPlantBuilder(const TailsitterParameters& params_)
      : params(params_) {}

  MultibodyPlant<double>* build(systems::DiagramBuilder<double>& builder,
                                SceneGraph<double>* scene_graph) const {
    /**
     * Tailsitter-axis: x-z plane with z axis along thrust direction, y-axis
     * normal to surface
     */
    auto plant = builder.AddSystem<MultibodyPlant<double>>(0.0);

    // ============== build wing =================
    // wing COM in wing frame
    const Vector3<double> p_WWcm(-params.wing_lz() / 2 *
                                 Vector3<double>::UnitZ());
    // inertia about COM
    auto i_Bcm = UnitInertia<double>::SolidBox(
        params.wing_lx(), params.wing_ly(), params.wing_lz());
    // inertia in wing frame, origin at head
    auto I_Wo = SpatialInertia<double>::MakeFromCentralInertia(
        params.wing_mass(), params.wing_lz() / 2 * Vector3<double>::UnitZ(),
        i_Bcm);
    const RigidBody<double>& wing = plant->AddRigidBody("wing", I_Wo);

    plant->RegisterAsSourceForSceneGraph(scene_graph);

    // wing visual
    const RigidTransformd X_WG(-params.wing_lz() / 2 *
                               Vector3<double>::UnitZ());
    plant->RegisterVisualGeometry(
        wing, X_WG, Box(params.wing_lx(), params.wing_ly(), params.wing_lz()),
        "wing_visual");

    // before we can add any plants(propeller, lifting surfaces), lets add
    // forcesMux. todo: fix after resolution of:
    // https://github.com/RobotLocomotion/drake/issues/13139
    auto forces_mux =
        builder.AddSystem<ExternalSpatialForceMultiplexer<double>>(
            std::vector<int>({1, 2}));

    // add wing-dynamics
    math::RigidTransform<double> X_BA;
    X_BA.set_translation(-params.wing_aerodynamic_center() *
                         Vector3<double>::UnitZ());
    // lift is along z axis for flatwing model
    X_BA.set_rotation(math::RotationMatrix<double>::MakeXRotation(M_PI_2));
    auto wing_info = FlatWingInfo(wing.index(), X_BA, params.wing_area());
    auto wing_dynamics = builder.AddSystem<FlatWing<double>>(
        std::vector<FlatWingInfo>({wing_info}), params.atmospheric_density());

    // ============== build elevon =================

    // ============== build props =================
    // todo: add mass effects of motors to intertia with
    // AddRigidBody 0: left, 1: right
    std::vector<multibody::PropellerInfo> props_info;
    for (int prop_i = 0; prop_i < 2; prop_i++) {
      math::RigidTransform<double> X_BP;
      X_BP.set_translation(pow(-1, prop_i) * params.propeller_com_distance() *
                           Vector3<double>::UnitX());
      props_info.push_back(
          PropellerInfo(wing.index(), X_BP, params.propeller_thrust_ratio(),
                        pow(-1, prop_i) * params.propeller_moment_ratio()));

      // prop visual
      plant->RegisterVisualGeometry(
          wing, X_BP,
          Cylinder(params.propeller_diameter() / 2, params.wing_ly()),
          fmt::format("prop_{}_visual", prop_i + 1));
    }
    auto props = builder.AddSystem<Propeller<double>>(props_info);

    // Gravity acting in the -z direction.
    plant->mutable_gravity_field().set_gravity_vector(-params.gravity() *
                                                      Vector3<double>::UnitZ());
    plant->Finalize();

    // connect wing dynamics port
    builder.Connect(wing_dynamics->get_spatial_forces_output_port(),
                    forces_mux->get_input_port(0));
    builder.Connect(plant->get_body_poses_output_port(),
                    wing_dynamics->get_body_poses_input_port());
    builder.Connect(plant->get_body_spatial_velocities_output_port(),
                    wing_dynamics->get_body_velocities_input_port());

    // connect propeller dynamics port
    builder.Connect(props->get_spatial_forces_output_port(),
                    forces_mux->get_input_port(1));
    builder.Connect(plant->get_body_poses_output_port(),
                    props->get_body_poses_input_port());

    // all forces to plant
    builder.Connect(forces_mux->get_spatial_forces_output_port(),
                    plant->get_applied_spatial_force_input_port());

    builder.ExportInput(props->get_command_input_port(), "prop_command");

    return plant;
  }
  void build_wing(MultibodyPlant<double>& plant) {}

  void build_elevons(MultibodyPlant<double>& plant) {}

  void build_props(MultibodyPlant<double>& plant) {}
};
}  // namespace tailsitter3d
}  // namespace examples
}  // namespace drake
