// InstinctLab single-input policy adapter (history_len=0, wheel velocity loop).

#include "rl_controller/fsm/FSMState_RLInstinctLab.h"
#include "rl_controller/common/timeMarker.h"

void FSMState_RLInstinctLab::update_forward()
{
  const long long interval = static_cast<long long>(rl_params_->time_interval * 1000000);
  while (threadRunning) {
    long long _start_time = getSystemTime();

    if (!stop_update_) {
      update_observations();
      // Single-input: feed only current obs, no history buffer.
      std::vector<std::vector<tensor_element_t>> input_datas;
      input_datas.push_back(eigenToVector(obs_vec_));
      action_vec_ = vectorToEigen(inferrer_->computeActions(input_datas));
      obs_.last_actions = action_vec_;
    }
    absoluteWait(_start_time, interval);
  }
  threadRunning = false;
}

void FSMState_RLInstinctLab::run()
{
  _data->low_cmd->zero();
  DVec<tensor_element_t> vel = d2f(_data->low_state->dq);

  for (int i = 0; i < rl_params_->num_actions; i++) {
    tensor_element_t action_scaled = action_vec_[i] * rl_params_->action_scales[i];
    bool is_wheel_joint =
      std::find(_data->params->wheel_indices.begin(),
                _data->params->wheel_indices.end(), i) !=
      _data->params->wheel_indices.end();
    tensor_element_t command =
      action_scaled + (tensor_element_t)rl_params_->default_joint_angles[i];

    if (is_wheel_joint) {
      // Software velocity loop:  tau = kd * (action_scale - dq)
      // Matches training ImplicitActuator (stiffness 0, damping).
      _data->low_cmd->kp(i) = 0.0;
      _data->low_cmd->kd(i) = 0.0;
      _data->low_cmd->qd(i) = 0.0;
      _data->low_cmd->qd_dot(i) = 0.0;
      _data->low_cmd->tau_cmd(i) =
        rl_params_->joint_kd[i] * (action_scaled - vel[i]);
    } else {
      // Leg joint: position PD
      _data->low_cmd->kp(i) = rl_params_->joint_kp[i];
      _data->low_cmd->kd(i) = rl_params_->joint_kd[i];
      _data->low_cmd->qd(i) = command;
      _data->low_cmd->qd_dot(i) = 0.0;
      _data->low_cmd->tau_cmd(i) = 0.0;
    }
  }
}
