"""Action Mod Trainer."""
from typing import Optional, Type

import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict

import action_mod_rllib as amr


def create_action_mod_trainer(tune_kwargs, modify_actions):
    """Register ActionModTrainer."""
    # Detect which policy
    complete_run_or_experiment = tune_kwargs['run_or_experiment']

    # Throw an error if the ActionMod version of the policy was not chosen
    action_mod_string = 'ActionMod'
    if complete_run_or_experiment[0:len(action_mod_string)] \
            != action_mod_string:
        raise Exception("'run_or_experiment' must be prefixed by 'ActionMod'")

    # Detect which policy was chosen
    run_or_experiment = complete_run_or_experiment[len(action_mod_string):]

    action_mod_tf_policy_cls, action_mod_torch_policy_cls = \
        amr.policy.create_action_mod_policy(run_or_experiment, modify_actions)

    def get_policy_class(config: TrainerConfigDict) -> \
            Optional[Type[Policy]]:
        if config['framework'] == 'torch':
            return action_mod_torch_policy_cls

        return action_mod_tf_policy_cls

    # Obtain the vanilla trainer class to extend
    trainer_cls = ray.tune.registry.get_trainable_cls(
        run_or_experiment)

    # Extend the trainer class with the new action mod policy
    action_mod_trainer_cls = trainer_cls.with_updates(
        default_policy=action_mod_tf_policy_cls,
        get_policy_class=get_policy_class)

    # Register the trainer
    ray.tune.registry.register_trainable(tune_kwargs['run_or_experiment'],
                                         action_mod_trainer_cls)
