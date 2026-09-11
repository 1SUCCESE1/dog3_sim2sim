// InstinctLab single-input policy adapter (history_len=0, wheel velocity loop).
#ifndef RL_CONTROLLER__FSM__FSMSTATE_RL_INSTINCTLAB_H_
#define RL_CONTROLLER__FSM__FSMSTATE_RL_INSTINCTLAB_H_

#include "rl_controller/fsm/FSMState_RL.h"

class FSMState_RLInstinctLab : public FSMState_RL
{
public:
  FSMState_RLInstinctLab(
    std::shared_ptr<ControlFSMData> data, RLParameters * rl_params, std::string stateName)
  : FSMState_RL(data, rl_params, stateName)
  {
  }
  virtual ~FSMState_RLInstinctLab() {}

protected:
  void update_forward() override;
  void run() override;
};

#endif  // RL_CONTROLLER__FSM__FSMSTATE_RL_INSTINCTLAB_H_
