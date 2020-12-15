"""Utils file to obtain policy classes and default config and mixins."""
from ray.rllib.policy import tf_policy, torch_policy


def _get_ppo_data():
    # pylint: disable=import-outside-toplevel
    from ray.rllib.agents.ppo import ppo_tf_policy
    from ray.rllib.agents.ppo import ppo_torch_policy

    policy_tf_cls = ppo_tf_policy.PPOTFPolicy
    setup_tf_before_init = ppo_tf_policy.setup_config
    tf_mixins = [tf_policy.LearningRateSchedule,
                 tf_policy.EntropyCoeffSchedule,
                 ppo_tf_policy.KLCoeffMixin,
                 ppo_tf_policy.ValueNetworkMixin]

    policy_torch_cls = ppo_torch_policy.PPOTorchPolicy
    setup_torch_before_init = ppo_tf_policy.setup_config
    torch_mixins = [torch_policy.LearningRateSchedule,
                    torch_policy.EntropyCoeffSchedule,
                    ppo_torch_policy.KLCoeffMixin,
                    ppo_torch_policy.ValueNetworkMixin]

    return policy_tf_cls, setup_tf_before_init, tf_mixins, \
        policy_torch_cls, setup_torch_before_init, torch_mixins


def get_policy_data(run_or_experiment):
    """Return tf/torch policy and other policy specific data to extend."""
    if run_or_experiment == 'PPO':
        return _get_ppo_data()

    return _get_ppo_data()
