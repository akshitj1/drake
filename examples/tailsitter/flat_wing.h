#pragma once

#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/tree/body.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace multibody {

struct FlatWingInfo {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FlatWingInfo);

  explicit FlatWingInfo(const BodyIndex& body_index_,
                        const math::RigidTransform<double>& X_BA_,
                        double surface_area_)
      : body_index(body_index_), X_BA(X_BA_), surface_area(surface_area_) {}

  /** The BodyIndex of a Body in the MultibodyPlant to which the propeller is
  attached.  The spatial forces will be applied to this body. */
  BodyIndex body_index;

  /** Pose of aerodynamic center(position and direction of surface normal ie.
   * lift direction) measured in the body frame B. @default is the identity
   * matrix. */
  math::RigidTransform<double> X_BA{};

  double surface_area{1.0};
};

template <typename T>
class FlatWing final : public systems::LeafSystem<T> {
  // Declare friendship to enable scalar conversion.
  template <typename U>
  friend class FlatWing;

 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FlatWing);
  const std::vector<FlatWingInfo> info;
  const double atmospheric_density;

  FlatWing(const std::vector<FlatWingInfo>& info_,
           const double atmospheric_density_)
      : systems::LeafSystem<T>(systems::SystemTypeTag<FlatWing>{}),
        info(info_),
        atmospheric_density(atmospheric_density_) {
    this->DeclareAbstractInputPort(
        "body_poses", Value<std::vector<math::RigidTransform<T>>>());

    this->DeclareAbstractInputPort("body_velocities",
                                   Value<std::vector<SpatialVelocity<T>>>());
    // todo: is abstract?
    // this->DeclareInputPort("propeller_downwash", systems::kVectorValued,
    //                        num_wings());

    this->DeclareAbstractOutputPort(
        "spatial_forces",
        std::vector<ExternallyAppliedSpatialForce<T>>(num_wings()),
        &FlatWing<T>::CalcSpatialForces);
  }

  /** Scalar-converting copy constructor.  See @ref system_scalar_conversion. */
  // todo: we are constructing wigth
  template <typename U>
  explicit FlatWing(const FlatWing<U>& other)
      : FlatWing<T>(
            std::vector<FlatWingInfo>(other.info.begin(), other.info.end()),
            other.atmospheric_density) {}

  int num_wings() const { return info.size(); }

  /** Returns a reference to the body_poses input port.  It is anticipated
  that this port will be connected the body_poses output port of a
  MultibodyPlant. */
  const systems::InputPort<T>& get_body_poses_input_port() const {
    return this->get_input_port(0);
  }

  const systems::InputPort<T>& get_body_velocities_input_port() const {
    return this->get_input_port(1);
  }

  // const systems::InputPort<T>& get_propeller_downwash_input_port() const {
  //   return this->get_input_port(2);
  // }

  /** Returns a reference to the spatial_forces output port.  It is anticipated
  that this port will be connected to the @ref
  MultibodyPlant::get_applied_spatial_force_input_port() "applied_spatial_force"
  input port of a MultibodyPlant. */
  const systems::OutputPort<T>& get_spatial_forces_output_port() const {
    return this->get_output_port(0);
  }

  void CalcSpatialForces(
      const systems::Context<T>& context,
      std::vector<ExternallyAppliedSpatialForce<T>>* spatial_forces) const {
    spatial_forces->resize(num_wings());

    // const auto& command = get_propeller_downwash_input_port().Eval(context);
    const auto& poses =
        get_body_poses_input_port()
            .template Eval<std::vector<math::RigidTransform<T>>>(context);

    const auto& vels =
        get_body_velocities_input_port()
            .template Eval<std::vector<SpatialVelocity<T>>>(context);

    for (int i = 0; i < num_wings(); i++) {
      const FlatWingInfo& wing = info[i];

      // Map to the ExternalSpatialForce structure:
      //  - the origin of my frame P is Po == Bq, and
      //  - the origin of my frame B is Bo.
      const math::RigidTransform<T>& X_WB = poses[wing.body_index];
      const Vector3<T>& p_BoAo_B = wing.X_BA.translation().cast<T>();
      const auto& vel_Bo_W = vels[wing.body_index];
      const auto& R_WB = X_WB.rotation();
      const auto& p_BoAo_W = R_WB * p_BoAo_B;
      const auto& vel_Ao_W = vel_Bo_W.Shift(p_BoAo_W);
      const auto& p_dot_Ao_W = vel_Ao_W.translational();
      const auto& X_WA = X_WB * wing.X_BA.cast<T>();
      const auto& R_WA = X_WA.rotation();

      // get angle of attack b/w wing plane(x-y) and its origin velocity
      // vector
      T sin_attack_angle =
          -p_dot_Ao_W.normalized().dot(R_WA * Vector3<T>::UnitZ());
      T aero_force_magnitude = atmospheric_density * p_dot_Ao_W.squaredNorm() *
                               wing.surface_area * sin_attack_angle;

      const SpatialForce<T> F_BAo_A(Vector3<T>::Zero(),
                                    aero_force_magnitude * Vector3<T>::UnitZ());

      const SpatialForce<T> F_BAo_W = R_WA * F_BAo_A;

      spatial_forces->at(i).body_index = wing.body_index;
      spatial_forces->at(i).p_BoBq_B = p_BoAo_B;
      spatial_forces->at(i).F_Bq_W = F_BAo_W;
    }
  }
};
}  // namespace multibody
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class drake::multibody::FlatWing)
