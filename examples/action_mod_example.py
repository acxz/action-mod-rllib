"""Script to run experiment."""
from typing import Dict, List, Optional

import ray
from ray.rllib.utils.typing import TensorType, Tuple, Union

import action_mod_rllib as amr


def train_cartpole():
    """Cartpole experiment."""
    ray.init()

    tune_kwargs = {
        # 'run_or_experiment': 'PPO',
        'run_or_experiment': 'ActionModPPO',
        'stop': {
            'episode_reward_mean': 150,
            'timesteps_total': 100000,
        },
        'config': {
            'env': 'CartPole-v0',
            'framework': 'torch',
            'gamma': 0.99,
            'lr': 0.0003,
            'num_workers': 1,
            'observation_filter': 'MeanStdFilter',
            'num_sgd_iter': 6,
            'vf_share_layers': True,
            'vf_loss_coeff': 0.01,
            'model': {
                'fcnet_hiddens': [32],
                'fcnet_activation': 'linear',
            }
        }
    }

    # pylint: disable=too-many-arguments
    # pylint: disable=unused-argument
    def modify_actions(
            obs_batch: Union[List[TensorType], TensorType],
            actions: TensorType,
            state_out: List[TensorType],
            extra_fetches: Dict[str, TensorType],
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

        # Modify the actions from the original computed ones
        # Or compute your own actions
        modified_actions = actions

        # Need to return these as well, don't worry about changing them unless
        # you know what you are doing
        modified_state_out = state_out
        modified_extra_fetches = extra_fetches

        return modified_actions, modified_state_out, modified_extra_fetches

    # Create the trainer that uses our overriden version of computing the
    # actions
    amr.trainer.create_action_mod_trainer(tune_kwargs, modify_actions)

    ray.tune.run(**tune_kwargs)


if __name__ == '__main__':
    train_cartpole()
