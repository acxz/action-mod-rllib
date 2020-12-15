"""Action Mod Policy."""
from typing import Dict, List, Optional

from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, Tuple, Union

import action_mod_rllib as amr


def make_action_mod_mixin(modify_actions):
    """Create mixin class based on given action modifier function."""
    # pylint: disable=too-few-public-methods
    class ActionModMixin:
        """Mixin class that overrides the action computation of the policy."""

        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        @override(Policy)
        def compute_actions(
                self,
                obs_batch: Union[List[TensorType], TensorType],
                state_batches: Optional[List[TensorType]] = None,
                prev_action_batch: Union[List[TensorType], TensorType] = None,
                prev_reward_batch: Union[List[TensorType], TensorType] = None,
                info_batch: Optional[Dict[str, list]] = None,
                episodes: Optional[List["MultiAgentEpisode"]] = None,
                explore: Optional[bool] = None,
                timestep: Optional[int] = None,
                **kwargs) -> \
                Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
            """Override the compute_actions of the policy."""
            actions, state_out, extra_fetches = \
                super().compute_actions(obs_batch, state_batches,
                                        prev_action_batch, prev_reward_batch,
                                        info_batch, episodes, explore,
                                        timestep, **kwargs)

            modified_actions, modified_state_out, modified_extra_fetches = \
                modify_actions(obs_batch, actions, state_out, extra_fetches,
                               state_batches, prev_action_batch,
                               prev_reward_batch, info_batch, episodes,
                               explore, timestep, **kwargs)

            return modified_actions, modified_state_out, modified_extra_fetches

    return ActionModMixin


def create_action_mod_policy(run_or_experiment, modify_actions):
    """Create an action modified policy."""
    # Obtain the default policy data of the vanilla trainer to extend
    policy_tf_cls, setup_tf_before_init, tf_mixins, \
        policy_torch_cls, setup_torch_before_init, torch_mixins = \
        amr.utils.get_policy_data(run_or_experiment)

    # Create the ActionModMixin
    ActionModMixin = make_action_mod_mixin(modify_actions)

    # pylint: disable=unused-argument
    def setup_action_mod_mixin_before_init(policy, obs_space, action_space,
                                           config):
        """Initialize ActionModMixin."""
        ActionModMixin.__init__(policy)

    def setup_action_mod_mixin_tf_before_init(policy, obs_space, action_space,
                                              config):
        """Initialize tf default and ActionMod Mixins."""
        setup_tf_before_init(policy, obs_space, action_space,
                             config)
        setup_action_mod_mixin_before_init(policy, obs_space, action_space,
                                           config)

    def setup_action_mod_mixin_torch_before_init(policy, obs_space,
                                                 action_space, config):
        """Initialize torch default and ActionMod Mixins."""
        setup_torch_before_init(policy, obs_space, action_space,
                                       config)
        setup_action_mod_mixin_before_init(policy, obs_space, action_space,
                                           config)

    # Extend existing mixins with the ActionModMixin
    tf_mixins.append(ActionModMixin)
    torch_mixins.append(ActionModMixin)

    # Extend the default policy with the extended mixins
    action_mod_tf_policy_cls = policy_tf_cls.with_updates(
        name='ActionModTFPolicy',
        before_init=setup_action_mod_mixin_tf_before_init,
        mixins=tf_mixins)

    action_mod_torch_policy_cls = policy_torch_cls.with_updates(
        name='ActionModTorchPolicy',
        before_init=setup_action_mod_mixin_torch_before_init,
        mixins=torch_mixins)

    return action_mod_tf_policy_cls, action_mod_torch_policy_cls
