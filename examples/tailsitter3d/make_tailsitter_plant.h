#pragma once

#include <memory>
#include <string>

#include "drake/examples/tailsitter3d/external_spatial_force_multiplexer.h"
#include "drake/examples/tailsitter3d/flat_wing.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/propeller.h"
#include "drake/multibody/tree/revolute_joint.h"

namespace drake {
namespace examples {
namespace tailsitter3d {
using drake::math::RigidTransformd;
using geometry::Box;
using geometry::Cylinder;
using geometry::SceneGraph;
using math::RigidTransform;
using math::RotationMatrix;
using multibody::ExternalSpatialForceMultiplexer;
using multibody::FlatWing;
using multibody::FlatWingInfo;
using multibody::MultibodyPlant;
using multibody::Propeller;
using multibody::PropellerInfo;
using multibody::RevoluteJoint;
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
  double elevon_lx() const { return elevon_lx_; }
  double elevon_ly() const { return elevon_ly_; }
  double elevon_lz() const { return elevon_lz_; }
  double wing_mass() const { return wing_mass_; }
  double elevon_mass() const { return elevon_mass_; }
  double tailsitter_mass() const { return 2 * elevon_mass_ + wing_mass_; }
  double propeller_com_distance() const { return propeller_com_distance_; }
  double propeller_diameter() const { return propeller_diameter_; }
  double propeller_thrust_ratio() const { return propeller_thrust_ratio_; }
  double propeller_moment_ratio() const { return propeller_moment_ratio_; }
  double wing_aerodynamic_center() const { return wing_aerodynamic_center_; }
  double wing_area() const { return wing_lx_ * wing_lz_; }
  double elevon_area() const { return elevon_lx_ * elevon_lz_; }
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

    plant->RegisterAsSourceForSceneGraph(scene_graph);

    std::vector<FlatWingInfo> lift_surfaces_info;

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

    // wing visual
    const RigidTransformd X_WG(-params.wing_lz() / 2 *
                               Vector3<double>::UnitZ());
    plant->RegisterVisualGeometry(
        wing, X_WG, Box(params.wing_lx(), params.wing_ly(), params.wing_lz()),
        "wing_visual");

    // add wing-dynamics
    RigidTransform<double> X_BA;
    X_BA.set_translation(-params.wing_aerodynamic_center() *
                         Vector3<double>::UnitZ());
    // lift is along z axis for flatwing model
    X_BA.set_rotation(math::RotationMatrix<double>::MakeXRotation(M_PI_2));
    lift_surfaces_info.push_back(
        FlatWingInfo(wing.index(), X_BA, params.wing_area()));

    // ============== build elevon =================
    for (int i = 0; i < 2; i++) {
      const std::string elevon_name =
          fmt::format("elevon_{}", i == 0 ? "left" : "right");
      // elevon COM in elevon frame
      const Vector3<double> p_EoEcm(-params.elevon_lz() / 2 *
                                    Vector3<double>::UnitZ());
      // inertia about COM
      auto i_Ecm = UnitInertia<double>::SolidBox(
          params.elevon_lx(), params.elevon_ly(), params.elevon_lz());
      // inertia in wing frame, origin at head
      auto I_Eo = SpatialInertia<double>::MakeFromCentralInertia(
          params.elevon_mass(),
          params.elevon_lz() / 2 * Vector3<double>::UnitZ(), i_Ecm);
      const RigidBody<double>& elevon = plant->AddRigidBody(elevon_name, I_Eo);

      // elevon visual
      const RigidTransform<double> X_EG(-params.elevon_lz() / 2 *
                                        Vector3<double>::UnitZ());
      plant->RegisterVisualGeometry(
          elevon, X_EG,
          Box(params.elevon_lx(), params.elevon_ly(), params.elevon_lz()),
          elevon_name + "_visual");

      // add joint
      const RigidTransform<double> X_wing_elevon(Vector3<double>(
          pow(-1, i % 2) * params.wing_lx() / 4, 0, -params.wing_lz()));
      const RevoluteJoint<double>& hinge = plant->AddJoint<RevoluteJoint>(
          elevon_name + "_joint", wing, X_wing_elevon, elevon,
          std::optional<RigidTransform<double>>{}, Vector3<double>::UnitX());

      // add elevon-dynamics
      math::RigidTransform<double> X_BA;
      X_BA.set_translation(-params.elevon_lz() / 2 * Vector3<double>::UnitZ());
      // lift is along z axis for flatwing model
      X_BA.set_rotation(RotationMatrix<double>::MakeXRotation(M_PI_2));
      lift_surfaces_info.push_back(
          FlatWingInfo(elevon.index(), X_BA, params.elevon_area()));
    }

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
    // Gravity acting in the -z direction.
    plant->mutable_gravity_field().set_gravity_vector(-params.gravity() *
                                                      Vector3<double>::UnitZ());
    plant->Finalize();

    // before we can add any plants(propeller, lifting surfaces), lets add
    // forcesMux. todo: fix after resolution of:
    // https://github.com/RobotLocomotion/drake/issues/13139
    std::vector<int> port_sizes;
    port_sizes.push_back(lift_surfaces_info.size());
    port_sizes.push_back(props_info.size());
    auto forces_mux =
        builder.AddSystem<ExternalSpatialForceMultiplexer<double>>(port_sizes);

    auto lift_surfaces = builder.AddSystem<FlatWing<double>>(
        lift_surfaces_info, params.atmospheric_density());

    auto props = builder.AddSystem<Propeller<double>>(props_info);

    // connect wing dynamics port
    builder.Connect(lift_surfaces->get_spatial_forces_output_port(),
                    forces_mux->get_input_port(0));
    builder.Connect(plant->get_body_poses_output_port(),
                    lift_surfaces->get_body_poses_input_port());
    builder.Connect(plant->get_body_spatial_velocities_output_port(),
                    lift_surfaces->get_body_velocities_input_port());

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
