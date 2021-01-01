#pragma once

#include <memory>
#include <string>

#include "drake/examples/tailsitter/external_spatial_force_multiplexer.h"
#include "drake/examples/tailsitter/flat_wing.h"
#include "drake/examples/tailsitter/tailsitter_parameters.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/propeller.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/revolute_spring.h"

namespace drake {
namespace examples {
namespace tailsitter {
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
using std::make_unique;
using std::string;
using std::unique_ptr;
using std::vector;

class TailsitterBuilder {
 private:
  static const RigidBody<double>& getAerodynamicBody(
      MultibodyPlant<double>* plant, const AeroSurfaceParameters& params,
      const string& name, vector<FlatWingInfo>& lift_surfaces_info) {
    // wing COM in wing frame
    const Vector3<double> p_WWcm(-params.dims()(2) / 2 *
                                 Vector3<double>::UnitZ());
    // inertia about COM
    auto i_Bcm = UnitInertia<double>::SolidBox(
        params.dims()(0), params.dims()(1), params.dims()(2));
    // inertia in wing frame, origin at head
    auto I_Wo = SpatialInertia<double>::MakeFromCentralInertia(
        params.mass(), params.dims()(2) / 2 * Vector3<double>::UnitZ(), i_Bcm);
    const RigidBody<double>& aero_body = plant->AddRigidBody(name, I_Wo);

    // wing visual
    const RigidTransformd X_WG(-params.dims()(2) / 2 *
                               Vector3<double>::UnitZ());
    plant->RegisterVisualGeometry(
        aero_body, X_WG,
        Box(params.dims()(0), params.dims()(1), params.dims()(2)),
        name + "_visual");

    // add wing-dynamics
    RigidTransform<double> X_BA;
    X_BA.set_translation(-params.center_of_pressure() *
                         Vector3<double>::UnitZ());
    // lift is along z axis for flat wing model
    X_BA.set_rotation(math::RotationMatrix<double>::MakeXRotation(M_PI_2));
    lift_surfaces_info.push_back(
        FlatWingInfo(aero_body.index(), X_BA, params.area()));

    return aero_body;
  }

  static const RigidBody<double>& getWing(
      MultibodyPlant<double>* plant, const AeroSurfaceParameters& wing_params,
      vector<FlatWingInfo>& lift_surfaces_info) {
    return getAerodynamicBody(plant, wing_params, "wing", lift_surfaces_info);
  }
  static const RigidBody<double>& getElevon(
      MultibodyPlant<double>* plant, const AeroSurfaceParameters& elevon_params,
      const string& name, vector<FlatWingInfo>& lift_surfaces_info) {
    return getAerodynamicBody(plant, elevon_params, name, lift_surfaces_info);
  }
  static RigidTransform<double> getElevonJointPose(
      int idx, const AeroSurfaceParameters& wing_params) {
    return RigidTransform<double>(
        Vector3<double>(pow(-1, idx % 2) * wing_params.dims()(0) / 4, 0,
                        -wing_params.dims()(2)));
  }
  static string getElevonName(int idx) {
    return fmt::format("elevon_{}", (idx == 0 ? "left" : "right"));
  }

  static string getElevonJointName(int idx) {
    return getElevonName(idx) + "_joint";
  }

  static const RevoluteJoint<double>& getElevonJoint(
      MultibodyPlant<double>* plant, const RigidBody<double>& wing,
      const RigidBody<double>& elevon, const string joint_name,
      const RigidTransform<double>& pose_wing_elevon,
      const double damping = 0.1, const double spring_stifness = 10.) {
    const auto joint = &plant->AddJoint<RevoluteJoint>(
        joint_name, wing, pose_wing_elevon, elevon,
        std::optional<RigidTransform<double>>{}, Vector3<double>::UnitX(),
        damping);

    plant->AddJointActuator("tau_" + joint_name, *joint);
    plant->AddForceElement<multibody::RevoluteSpring>(*joint, 0.,
                                                      spring_stifness);

    return *joint;
  }

  /**
   * adds elevon and wing with aerodynamics
   */
  static unique_ptr<FlatWing<double>> getLiftSurfaces(
      MultibodyPlant<double>* plant, const TailsitterParameters& params) {
    vector<FlatWingInfo> lift_surfaces_info;
    const auto& wing =
        getAerodynamicBody(plant, params.wing(), "wing", lift_surfaces_info);
    for (int i = 0; i < 2; i++) {
      const auto& elevon = getAerodynamicBody(
          plant, params.elevon(), getElevonName(i), lift_surfaces_info);
      // const auto& elevon_joint =
      getElevonJoint(plant, wing, elevon, getElevonJointName(i),
                     getElevonJointPose(i, params.wing()));
    }
    return make_unique<FlatWing<double>>(lift_surfaces_info,
                                         params.world().atmospheric_density());
  }

  static unique_ptr<Propeller<double>> getPropellers(
      MultibodyPlant<double>* plant, const TailsitterParameters& params,
      const RigidBody<double>& wing) {
    // todo: add mass effects of motors to intertia with
    std::vector<multibody::PropellerInfo> props_info;
    for (int prop_i = 0; prop_i < 2; prop_i++) {
      math::RigidTransform<double> X_BP;
      X_BP.set_translation(pow(-1, prop_i) * params.propeller_com_distance() *
                           Vector3<double>::UnitX());
      props_info.push_back(
          PropellerInfo(wing.index(), X_BP, params.propeller().thrust_ratio(),
                        pow(-1, prop_i) * params.propeller().moment_ratio()));

      // prop visual
      plant->RegisterVisualGeometry(
          wing, X_BP,
          Cylinder(params.propeller().diameter() / 2, params.wing().dims()(1)),
          fmt::format("prop_{}_visual", prop_i + 1));
    }
    return make_unique<Propeller<double>>(props_info);
  }

 public:
  static const MultibodyPlant<double>* getPlant(
      const TailsitterParameters& params,
      systems::DiagramBuilder<double>& builder,
      SceneGraph<double>* scene_graph) {
    auto plant = builder.AddSystem<MultibodyPlant<double>>(0.0);
    plant->RegisterAsSourceForSceneGraph(scene_graph);

    auto lift_surfaces = getLiftSurfaces(plant, params);
    auto props =
        getPropellers(plant, params, plant->GetRigidBodyByName("wing"));
    plant->Finalize();

    // before we can add any plants(propeller, lifting surfaces), lets
    // add forcesMux. todo: fix after resolution of:
    // https://github.com/RobotLocomotion/drake/issues/13139
    vector<unsigned int> port_sizes{3, 2};
    // port_sizes.push_back(lift_surfaces_info.size());
    // port_sizes.push_back(props_info.size());
    auto forces_mux =
        builder.AddSystem<ExternalSpatialForceMultiplexer<double>>(port_sizes);

    auto lift_systems = builder.AddSystem(std::move(lift_surfaces));
    auto prop_systems = builder.AddSystem(std::move(props));

    // connect wing dynamics port
    builder.Connect(lift_systems->get_spatial_forces_output_port(),
                    forces_mux->get_input_port(0));
    builder.Connect(plant->get_body_poses_output_port(),
                    lift_systems->get_body_poses_input_port());
    builder.Connect(plant->get_body_spatial_velocities_output_port(),
                    lift_systems->get_body_velocities_input_port());

    // connect propeller dynamics port
    builder.Connect(prop_systems->get_spatial_forces_output_port(),
                    forces_mux->get_input_port(1));
    builder.Connect(plant->get_body_poses_output_port(),
                    prop_systems->get_body_poses_input_port());

    // all forces to plant
    builder.Connect(forces_mux->get_spatial_forces_output_port(),
                    plant->get_applied_spatial_force_input_port());

    builder.ExportInput(prop_systems->get_command_input_port(), "prop_command");

    return plant;
  }

  static const MultibodyPlant<double>* getPlant(
      systems::DiagramBuilder<double>& builder,
      SceneGraph<double>* scene_graph) {
    return getPlant(TailsitterParameters(), builder, scene_graph);
  }
};
}  // namespace tailsitter
}  // namespace examples
}  // namespace drake
