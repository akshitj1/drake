#pragma once

#include <memory>

#include "drake/examples/tailsitter/gen/tailsitter_state.h"
#include "drake/examples/tailsitter/tailsitter_plant.h"
#include "drake/geometry/geometry_frame.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/geometry_instance.h"
#include "drake/geometry/geometry_roles.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace tailsitter {

using Eigen::Vector3d;
using Eigen::Vector4d;
using geometry::Box;
using geometry::Cylinder;
using geometry::GeometryFrame;
using geometry::GeometryId;
using geometry::GeometryInstance;
using geometry::MakePhongIllustrationProperties;
using geometry::Sphere;
using std::make_unique;

/// Expresses an TailsitterPlant's geometry to a SceneGraph.
///
/// @system{TailsitterGeometry,
///    @input_port{state},
///    @output_port{geometry_pose}
/// }
///
/// This class has no public constructor; instead use the AddToBuilder() static
/// method to create and add it to a DiagramBuilder directly.
class TailsitterGeometry final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TailsitterGeometry);
  //   ~TailsitterGeometry() final;
  ~TailsitterGeometry() = default;

  static const TailsitterGeometry* AddToBuilder(
      systems::DiagramBuilder<double>* builder,
      const systems::OutputPort<double>& tailsitter_state_port,
      geometry::SceneGraph<double>* scene_graph) {
    DRAKE_THROW_UNLESS(builder != nullptr);
    DRAKE_THROW_UNLESS(scene_graph != nullptr);

    auto tailsitter_geometry =
        builder->AddSystem(std::unique_ptr<TailsitterGeometry>(
            new TailsitterGeometry(scene_graph)));
    builder->Connect(tailsitter_state_port,
                     tailsitter_geometry->get_input_port(0));
    builder->Connect(
        tailsitter_geometry->get_output_port(0),
        scene_graph->get_source_pose_port(tailsitter_geometry->source_id));

    return tailsitter_geometry;
  }

 private:
  // Geometry source identifier for this system to interact with SceneGraph.
  geometry::SourceId source_id{};
  // The frames for the two links.
  geometry::FrameId wing_link_id{};
  geometry::FrameId elevon_link_id{};
  geometry::FrameId prop_link_id{};

  const double elevon_height = 2 * Tailsitter<double>::kLe,
               elevon_width = Tailsitter<double>::kTailS / elevon_height,
               wing_width = elevon_width,
               wing_height = Tailsitter<double>::kWingS / wing_width,
               tailsitter_thick = 0.005;  // 5mm

  TailsitterGeometry(geometry::SceneGraph<double>* scene_graph) {
    DRAKE_THROW_UNLESS(scene_graph != nullptr);
    source_id = scene_graph->RegisterSource();

    this->DeclareVectorInputPort("state", TailsitterState<double>());
    this->DeclareAbstractOutputPort("geometry_pose",
                                    &TailsitterGeometry::OutputGeometryPose);

    GeometryId id;

    wing_link_id =
        scene_graph->RegisterFrame(source_id, GeometryFrame("wing_link"));

    id = scene_graph->RegisterGeometry(
        source_id, wing_link_id,
        make_unique<GeometryInstance>(
            math::RigidTransformd(Vector3d(0., 0., 0.)),
            make_unique<Box>(wing_height, wing_width, tailsitter_thick),
            "wing_link"));
    scene_graph->AssignRole(
        source_id, id, MakePhongIllustrationProperties(Vector4d(1, 0, 0, 1)));

    elevon_link_id = scene_graph->RegisterFrame(source_id, wing_link_id,
                                                GeometryFrame("elevon_link"));
    id = scene_graph->RegisterGeometry(
        source_id, elevon_link_id,
        make_unique<GeometryInstance>(
            math::RigidTransformd(Vector3d(elevon_height / 2, 0., 0.)),
            make_unique<Box>(elevon_height, elevon_width, tailsitter_thick),
            "elevon_link"));
    scene_graph->AssignRole(
        source_id, id, MakePhongIllustrationProperties(Vector4d(0, 1, 0, 1)));

    prop_link_id = scene_graph->RegisterFrame(source_id, wing_link_id,
                                              GeometryFrame("prop_link"));
    id = scene_graph->RegisterGeometry(
        source_id, prop_link_id,
        make_unique<GeometryInstance>(
            math::RigidTransformd(math::RotationMatrixd::MakeYRotation(M_PI_2),
                                  Vector3d(-wing_height / 2, 0., 0.)),
            make_unique<Cylinder>(Tailsitter<double>::kPropDiameter / 2,
                                  tailsitter_thick),
            "prop_link"));
    scene_graph->AssignRole(
        source_id, id, MakePhongIllustrationProperties(Vector4d(0, 0, 1, 0.5)));
  }

  void OutputGeometryPose(const systems::Context<double>& context,
                          geometry::FramePoseVector<double>* poses) const {
    DRAKE_DEMAND(wing_link_id.is_valid());
    DRAKE_DEMAND(elevon_link_id.is_valid());

    const auto& state =
        get_input_port(0).Eval<TailsitterState<double>>(context);
    const math::RigidTransformd wing_pose(
        math::RotationMatrixd::MakeYRotation(state.theta()),
        Vector3d(state.x(), 0, state.z()));

    const math::RigidTransformd elevon_pose(
        math::RotationMatrixd::MakeYRotation(state.phi()),
        Vector3d(wing_height / 2, 0, 0));

    const math::RigidTransformd prop_pose;

    *poses = {{wing_link_id, wing_pose},
              {elevon_link_id, elevon_pose},
              {prop_link_id, prop_pose}};
  }
  static Vector2<double> rotate(const Vector2<double>& x, const double& theta) {
    return Vector2<double>(x(0) * cos(theta) - x(1) * sin(theta),
                           x(0) * sin(theta) + x(1) * cos(theta));
  }
};

}  // namespace tailsitter
}  // namespace examples
}  // namespace drake
