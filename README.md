# action-mod-rllib

Override [rllib](https://docs.ray.io/en/master/rllib.html) policies with
your own action modification.

## Usage

There are 3 things to do to use this library:

1. Create a function which performs your action modification.

- See https://github.com/acxz/action-mod-rllib/blob/master/examples/action_mod_example.py#L40
for an example `modify_action` function.

2. Changes to `tune_kwargs`:

- Change the `run_or_experiment` entry to `ActionMod<run_or_experiment>`.

  For example, if you want to override PPO's action computation, instead of having

  ```python
  'run_or_experiment': 'PPO'
  ```

  use the following:
  ```python
  'run_or_experiment': 'ActionModPPO'
  ```

- Add the following entry:

  ```python
  'modify_action': <modify_action_method>
  ```

  For example if your method is `modify_action`:
  ```python
  `modify_action`: modify_action
  ```

3. Create a trainer which will utilize the `modify_action` method:

    ```python
    action_mod_rllib.trainer.create_action_mod_trainer(tune_kwargs)
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
