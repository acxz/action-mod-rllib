# action-mod-rllib

Override [rllib](https://docs.ray.io/en/master/rllib.html) policies with
your own action modification.

## Usage

There are 3 things to do to use this library:

1. Change the `run_or_experiment` entry in `tune_kwargs` to
`ActionMod<run_or_experiment>`.

    For example, if you want to override PPO's action computation, instead of
    having

    ```python
    'run_or_experiment': 'PPO'
    ```

    use the following:
    ```python
    'run_or_experiment': 'ActionModPPO'
    ```

2. Create a function which performs your action modification.

    See [here](https://github.com/acxz/action-mod-rllib/blob/master/examples/action_mod_example.py)
    for an example `modify_actions` function.

3. Create a trainer which will utilize the `my_modify_actions` method:

    ```python
    action_mod_rllib.trainer.create_action_mod_trainer(tune_kwargs, my_modify_actions)
    ```

Example usage is in `examples/`.

```bash
python examples/action_mod_example.py
```

## Installation

Dependencies will be pulled in automatically by `pip`.

To install this package run the following command within this directory:

```bash
pip install .
```

## Uninstall

To uninstall the package:

```bash
pip uninstall action-mod-rllib
```
