import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common.logger import Logger

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None


from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize, sync_envs_normalization

import plotly.graph_objects as go
from pathlib import Path
from collections import deque

if TYPE_CHECKING:
    from stable_baselines3.common import base_class


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    # The RL model
    # Type hint as string to avoid circular import
    model: "base_class.BaseAlgorithm"

    def __init__(self, verbose: int = 0):
        super().__init__()
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    @property
    def training_env(self) -> VecEnv:
        training_env = self.model.get_env()
        assert (
            training_env is not None
        ), "`model.get_env()` returned None, you must initialize the model with an environment to use callbacks"
        return training_env

    @property
    def logger(self) -> Logger:
        return self.model.logger

    # Type hint as string to avoid circular import
    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        :param locals_: the local variables during rollout collection
        """
        pass


class EventCallback(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: Callback that will be called
        when an event is triggered.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            assert self.callback is not None
            self.callback.parent = self

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        super().init_callback(model)
        if self.callback is not None:
            self.callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        if self.callback is not None:
            self.callback.on_training_start(self.locals, self.globals)

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback.on_step()
        return True

    def _on_step(self) -> bool:
        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback is not None:
            self.callback.update_locals(locals_)


class CallbackList(BaseCallback):
    """
    Class for chaining callbacks.

    :param callbacks: A list of callbacks that will be called
        sequentially.
    """

    def __init__(self, callbacks: List[BaseCallback]):
        super().__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)

            # Fix for https://github.com/DLR-RM/stable-baselines3/issues/1791
            # pass through the parent callback to all children
            callback.parent = self.parent

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        for callback in self.callbacks:
            callback.update_locals(locals_)


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True


class ConvertCallback(BaseCallback):
    """
    Convert functional callback (old-style) to object.

    :param callback:
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]], verbose: int = 0):
        super().__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)
        return True


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: List[List[bool]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

        # For logging
        self._logging_callback = EvalLoggingCallback(log_dir=self.logger.dir)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _eval_step_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """ """
        self._log_success_callback(locals_, globals_)

        step_infos = locals_["infos"]
        if isinstance(locals_["env"], VecNormalize):
            step_obs = locals_["env"].get_original_obs()
        else:
            step_obs = locals_["new_observations"]

        reward = locals_["rewards"]
        done = locals_["dones"]

        self._logging_callback.on_step([step_obs], step_infos, reward, done)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._eval_step_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    """

    parent: EvalCallback

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used with an ``EvalCallback``"
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
        return continue_training


class EveryNTimesteps(EventCallback):
    """
    Trigger a callback every ``n_steps`` timesteps

    :param n_steps: Number of timesteps between two trigger.
    :param callback: Callback that will be called
        when the event is triggered.
    """

    def __init__(self, n_steps: int, callback: BaseCallback):
        super().__init__(callback)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            return self._on_event()
        return True


class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Stop the training once a maximum number of episodes are played.

    For multiple environments presumes that, the desired behavior is that the agent trains on each env for ``max_episodes``
    and in total for ``max_episodes * n_envs`` episodes.

    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about when training ended by
        reaching ``max_episodes``
    """

    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0

    def _init_callback(self) -> None:
        # At start set total max according to number of environments
        self._total_max_episodes = self.max_episodes * self.training_env.num_envs

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()

        continue_training = self.n_episodes < self._total_max_episodes

        if self.verbose >= 1 and not continue_training:
            mean_episodes_per_env = self.n_episodes / self.training_env.num_envs
            mean_ep_str = (
                f"with an average of {mean_episodes_per_env:.2f} episodes per env" if self.training_env.num_envs > 1 else ""
            )

            print(
                f"Stopping training with a total of {self.num_timesteps} steps because the "
                f"{self.locals.get('tb_log_name')} model reached max_episodes={self.max_episodes}, "
                f"by playing for {self.n_episodes} episodes "
                f"{mean_ep_str}"
            )
        return continue_training


class StopTrainingOnNoModelImprovement(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.

    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

    It must be used with the ``EvalCallback``.

    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
    """

    parent: EvalCallback

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        continue_training = True

        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        self.last_best_mean_reward = self.parent.best_mean_reward

        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
            )

        return continue_training


class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    pbar: tqdm

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()


class EvalLoggingCallback:
    def __init__(
        self,
        vars_to_log: dict = {},
        vars_to_plot: list = [],
        verbose=0,
        stats_window_size=100,
        exclude_rolling_mean: list = [],
        plot_log_interval=2,
        track: bool = False,
        is_eval: bool = False,  # A bit confusing. If True, the callback is used from the enjoy script. Else from the EvalCallback
        log_dir: str = None,
    ):
        self.stats_window_size = stats_window_size

        self.is_eval = is_eval
        self.log_dir = log_dir
        if is_eval:
            self.plots_dir = Path(self.log_dir) / "plots"
        else:
            self.plots_dir = Path(self.log_dir) / "plots" / "eval"

        self.ep_num = 0  # Number of completed episodes
        self.plot_log_interval = plot_log_interval  # Frequency at which to log the plots

        self.num_envs = None
        self.vars_to_log = vars_to_log  # keys: variable names, values: "mean" or "last" (how to proc the values over the ep)
        self.vars_to_plot = vars_to_plot  # Variables to plot
        self.vars_to_plot_data = None  # Data to plot

        self.vars_to_plot = [
            # Obs
            "joint_angles",
            "joint_vels",
            "joint_torques",
            "joint_target_vels",
            # Infos
            "action",
            "reward",
            "joint_vels_ideal",
            "ee_vel_ctrl",
            "ee_vel_aug",
            "ee_vel",
        ]
        if is_eval:
            self.vars_to_eval = {
                "action": ["sum"],
                "joint_torques": ["sum"],
                "peg_force_norm": ["mean", "rms", "max"],
                "peg_torque_norm": ["mean", "rms", "max"],
                "socket_x": ["last"],
                "peg_pose_z": ["last"],
            }
        else:
            self.vars_to_eval = {}

        # Tracking
        self.track = track  # Whether to track the variables with wandb
        self.wandb_run = None
        self.episode_plots_data = []

        # Vars to log
        self.exclude_rolling_mean = exclude_rolling_mean  # Variables for which to exclude the rolling mean
        self.vars_values = None  # Store the values of the variables to log. {var_name: [[ts_1, ts_2], [ts_1, ...], ...]} where ts_i is the value of the variable at time step i. 1st dimension for env_idx

        self.vars_proc_values = None  # Store the processed values of the variables to log. {var_name: [ep_1, ep_2, ...]} where proc_val_i is the processed value of the variable for episode i

        self.eval_vars_values = None  # See vars_values
        self.eval_vars_proc_values = None  # See vars_proc_values

        self._init_vals()
        ...

    def _init_vals(self) -> None:
        # Initialize the variables to log
        self.num_envs = 1

        # LOG
        self.vars_values = {var_name: [[] for _ in range(self.num_envs)] for var_name in self.vars_to_log}

        self.vars_proc_values = {var_name: [] for var_name in self.vars_to_log}
        self.vars_rolling_values = {
            var_name: deque(maxlen=self.stats_window_size)
            for var_name in self.vars_to_log
            if var_name not in self.exclude_rolling_mean
        }

        # PLOT
        self.vars_to_plot_data = {var_name: [[] for _ in range(self.num_envs)] for var_name in self.vars_to_plot}

        # EVAL
        self.eval_vars_values = {}
        self.eval_vars_proc_values = {}
        for each_eval_var, eval_types in self.vars_to_eval.items():
            self.eval_vars_values[each_eval_var] = [[] for _ in range(self.num_envs)]
            self.eval_vars_proc_values[each_eval_var] = {proc_type: [] for proc_type in eval_types}
            # Example: eval_vars_values (reset after each episode)
            # #          # step 1   #step2
            # 'action': [[[0, 0.4], [0, 1], [1, 1], [0.2, 0.3]], [env_2]]
            #
            # Example: eval_vars_proc_values, value appended at each episode
            # 'action': {
            #     #       ep 1           # ep2
            #     #       #env1, env2
            #     'sum': [[0.4, 1, ...], [...]],
            # }

    def on_step(self, step_obs, step_infos, reward, done) -> bool:
        """Exract the variables to log from the step infos and observations and log them.

        Args:
            step_obs: np.ndarray(dict), observations of the env. Size of num_envs
            step_infos: np.ndarray(dict), info of the env. Size of num_envs
            reward: np.ndarray(dict), reward of the env. Size of num_envs
            done: np.ndarray(dict), whether the episode is done. Size of num_envs
        """
        for env_idx in range(self.num_envs):
            env_obs = step_obs[env_idx]
            env_infos = step_infos[env_idx]
            env_reward = reward[env_idx]
            env_done = done[env_idx]

            # LOG
            for var_name in self.vars_values.keys():
                var_value = env_infos.get(var_name)

                if var_value is not None:
                    self.vars_values[var_name][env_idx].append(var_value)

            # PLOT
            for var_name in self.vars_to_plot_data.keys():
                info_keys = env_infos.keys()
                obs_keys = env_obs.keys()
                if var_name in info_keys:
                    var_value = env_infos.get(var_name)
                elif var_name in obs_keys:
                    var_value = env_obs.get(var_name)
                elif var_name == "reward":
                    var_value = env_reward
                else:
                    raise ValueError(f"Variable {var_name} not found in env_infos or env_obs.")

                if var_value is not None:
                    self.vars_to_plot_data[var_name][env_idx].append(var_value)

            # EVAL
            for var_name in self.vars_to_eval.keys():
                info_keys = env_infos.keys()
                obs_keys = env_obs.keys()
                if var_name in info_keys:
                    var_value = env_infos.get(var_name)
                elif var_name in obs_keys:
                    var_value = env_obs.get(var_name)
                elif var_name == "reward":
                    var_value = env_reward
                else:
                    raise ValueError(f"Variable {var_name} not found in env_infos or env_obs.")

                if var_value is not None:
                    self.eval_vars_values[var_name][env_idx].append(var_value)

            # Check if the episode is done
            if env_done:
                self.ep_num += 1
                for var_name in self.vars_values.keys():
                    if self.vars_to_log[var_name] == "last":
                        proc_var = self.vars_values[var_name][env_idx][-1]
                    elif self.vars_to_log[var_name] == "mean":
                        proc_var = np.mean(self.vars_values[var_name][env_idx])
                    else:
                        raise ValueError(
                            f"Invalid proc type ({self.vars_to_log[var_name]}) for var: {var_name}. Must be 'mean' or 'last'."
                        )

                    self.vars_proc_values[var_name].append(proc_var)
                    # self.logger.record(f"custom/{var_name}", proc_var)

                    # Rolling
                    if var_name not in self.exclude_rolling_mean:
                        self.vars_rolling_values[var_name].append(proc_var)
                        rolling_mean = np.mean(self.vars_rolling_values[var_name])

                        # self.logger.record(f"custom/rolling_mean_{var_name}", rolling_mean)

                    # Reset the list of values for the variable
                    self.vars_values[var_name][env_idx] = []

                self._process_eval_vars(env_idx)

                if self.ep_num % self.plot_log_interval == 0 or self.ep_num == 1:
                    self.plot_episode(env_idx)

                for var_name in self.vars_to_plot_data.keys():
                    self.vars_to_plot_data[var_name][env_idx] = []

        return True

    def plot_episode(self, env_idx) -> None:
        for var_name, var_values in self.vars_to_plot_data.items():
            data = np.array(var_values[env_idx])

            fig = go.Figure()

            # Add Trace depending on var_name
            if var_name == "joint_angles":
                fig.add_trace(go.Scatter(y=data[:-1, 0, 0], mode="lines+markers", name="j2"))
                fig.add_trace(go.Scatter(y=data[:-1, 0, 1], mode="lines+markers", name="j4"))
                fig.add_trace(go.Scatter(y=data[:-1, 0, 2], mode="lines+markers", name="j6"))
                y_label = "Angle [deg]"

            elif var_name == "joint_vels":
                fig.add_trace(go.Scatter(y=data[:-1, 0, 0], mode="lines+markers", name="j2"))
                fig.add_trace(go.Scatter(y=data[:-1, 0, 1], mode="lines+markers", name="j4"))
                fig.add_trace(go.Scatter(y=data[:-1, 0, 2], mode="lines+markers", name="j6"))
                y_label = "[deg/s]"

            elif var_name == "joint_target_vels":
                fig.add_trace(go.Scatter(y=data[:-1, 0, 0], mode="lines+markers", name="j2"))
                fig.add_trace(go.Scatter(y=data[:-1, 0, 1], mode="lines+markers", name="j4"))
                fig.add_trace(go.Scatter(y=data[:-1, 0, 2], mode="lines+markers", name="j6"))
                y_label = "[deg/s]"

            elif var_name == "joint_torques":
                fig.add_trace(go.Scatter(y=data[:-1, 0, 0], mode="lines+markers", name="j2"))
                fig.add_trace(go.Scatter(y=data[:-1, 0, 1], mode="lines+markers", name="j4"))
                fig.add_trace(go.Scatter(y=data[:-1, 0, 2], mode="lines+markers", name="j6"))
                y_label = "[N*m]"

            elif var_name == "action":
                fig.add_trace(go.Scatter(y=data[:-1, 0], mode="lines+markers", name="action_j2"))
                fig.add_trace(go.Scatter(y=data[:-1, 1], mode="lines+markers", name="action_j4"))
                try:
                    fig.add_trace(go.Scatter(y=data[:-1, 2], mode="lines+markers", name="action_j6"))
                except:
                    ...
                y_label = "Value"

            elif var_name == "joint_cmd":
                fig.add_trace(go.Scatter(y=data[:-1, 0], mode="lines+markers", name="j2"))
                fig.add_trace(go.Scatter(y=data[:-1, 1], mode="lines+markers", name="j4"))
                fig.add_trace(go.Scatter(y=data[:-1, 2], mode="lines+markers", name="j6"))
                y_label = "[deg/s]"

            elif var_name == "joint_vels_ideal":
                fig.add_trace(go.Scatter(y=data[:-1, 0], mode="lines+markers", name="j2"))
                fig.add_trace(go.Scatter(y=data[:-1, 1], mode="lines+markers", name="j4"))
                fig.add_trace(go.Scatter(y=data[:-1, 2], mode="lines+markers", name="j6"))
                y_label = "[deg/s]"

            elif var_name == "reward":
                fig.add_trace(go.Scatter(y=data[:-1], mode="lines+markers", name="reward"))
                y_label = "reward"

            elif var_name == "ee_vel_ctrl":
                fig.add_trace(go.Scatter(y=data[:-1, 0], mode="lines+markers", name="x"))
                fig.add_trace(go.Scatter(y=data[:-1, 1], mode="lines+markers", name="z"))
                fig.add_trace(go.Scatter(y=data[:-1, 2], mode="lines+markers", name="rot"))
                y_label = "[m/s]"

            elif var_name == "ee_vel_aug":
                fig.add_trace(go.Scatter(y=data[:-1, 0], mode="lines+markers", name="x"))
                fig.add_trace(go.Scatter(y=data[:-1, 1], mode="lines+markers", name="z"))
                fig.add_trace(go.Scatter(y=data[:-1, 2], mode="lines+markers", name="rot"))
                y_label = "[m/s]"

            elif var_name == "ee_vel":
                fig.add_trace(go.Scatter(y=data[:-1, 0], mode="lines+markers", name="x"))
                fig.add_trace(go.Scatter(y=data[:-1, 1], mode="lines+markers", name="z"))
                fig.add_trace(go.Scatter(y=data[:-1, 2], mode="lines+markers", name="rot"))
                y_label = "[m/s]"

            else:
                raise ValueError(f"Variable {var_name} plot not specified")

            fig.update_layout(
                title=f"Episode {self.ep_num} - {var_name.capitalize()}",
                xaxis_title="Time step",
                yaxis_title=y_label,
            )
            save_path = self.plots_dir / f"ep_{self.ep_num}" / f"{var_name}_{env_idx}.html"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path, auto_play=False)

        # Plot command, vels, and ctrl_cmd
        if (
            "joint_cmd" in self.vars_to_plot_data.keys()
            and "joint_vels_ideal" in self.vars_to_plot_data.keys()
            and "joint_vels" in self.vars_to_plot_data.keys()
        ):
            fig = go.Figure()
            joint_vels_data = np.array(self.vars_to_plot_data["joint_vels"][env_idx])
            joint_command_data = np.array(self.vars_to_plot_data["joint_cmd"][env_idx])
            joint_vels_ideal_data = np.array(self.vars_to_plot_data["joint_vels_ideal"][env_idx])

            # Joint command: What is sent to Vortex
            fig.add_trace(
                go.Scatter(
                    y=joint_command_data[:-1, 0],
                    mode="lines+markers",
                    name="j2_command",
                    line=dict(color="red", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=joint_command_data[:-1, 1],
                    mode="lines+markers",
                    name="j4_command",
                    line=dict(color="blue", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=joint_command_data[:-1, 2],
                    mode="lines+markers",
                    name="j6_command",
                    line=dict(color="green", dash="dash"),
                )
            )

            # joint_vels_ideal: Expected trajectory
            fig.add_trace(
                go.Scatter(y=joint_vels_ideal_data[:-1, 0], mode="lines+markers", name="j2_id", line=dict(color="red"))
            )
            fig.add_trace(
                go.Scatter(y=joint_vels_ideal_data[:-1, 1], mode="lines+markers", name="j4_id", line=dict(color="blue"))
            )
            fig.add_trace(
                go.Scatter(y=joint_vels_ideal_data[:-1, 2], mode="lines+markers", name="j6_id", line=dict(color="green"))
            )

            # joint_velocities: Actual velocities
            fig.add_trace(
                go.Scatter(
                    y=joint_vels_data[:-1, 0, 0],
                    mode="lines+markers",
                    name="j2_vel",
                    line=dict(color="red", dash="dashdot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=joint_vels_data[:-1, 0, 1],
                    mode="lines+markers",
                    name="j4_vel",
                    line=dict(color="blue", dash="dashdot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=joint_vels_data[:-1, 0, 2],
                    mode="lines+markers",
                    name="j6_vel",
                    line=dict(color="green", dash="dashdot"),
                )
            )

            plot_name = "joint_vels_comp"
            fig.update_layout(
                title=f"Episode {self.ep_num} - Comp",
                xaxis_title="Time step",
                yaxis_title=y_label,
            )
            save_path = self.plots_dir / f"ep_{self.ep_num}" / f"{plot_name}_{env_idx}.html"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path, auto_play=False)

        if (
            "ee_vel_ctrl" in self.vars_to_plot_data.keys()
            and "ee_vel_aug" in self.vars_to_plot_data.keys()
            and "ee_vel" in self.vars_to_plot_data.keys()
        ):
            fig = go.Figure()
            ee_vel_ctrl = np.array(self.vars_to_plot_data["ee_vel_ctrl"][env_idx])
            ee_vel_aug = np.array(self.vars_to_plot_data["ee_vel_aug"][env_idx])
            ee_vel = np.array(self.vars_to_plot_data["ee_vel"][env_idx])

            # ee_vel_ctrl
            fig.add_trace(
                go.Scatter(y=ee_vel_ctrl[:-1, 0], mode="lines+markers", name="ee_x_ctrl", line=dict(color="red", dash="dash"))
            )
            fig.add_trace(
                go.Scatter(
                    y=ee_vel_ctrl[:-1, 1],
                    mode="lines+markers",
                    name="ee_z_ctrl",
                    line=dict(color="blue", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=ee_vel_ctrl[:-1, 2],
                    mode="lines+markers",
                    name="ee_rot_ctrl",
                    line=dict(color="green", dash="dash"),
                )
            )

            # ee_vel_aug
            fig.add_trace(go.Scatter(y=ee_vel_aug[:-1, 0], mode="lines+markers", name="ee_x_aug", line=dict(color="red")))
            fig.add_trace(go.Scatter(y=ee_vel_aug[:-1, 1], mode="lines+markers", name="ee_z_aug", line=dict(color="blue")))
            fig.add_trace(go.Scatter(y=ee_vel_aug[:-1, 2], mode="lines+markers", name="ee_rot_aug", line=dict(color="green")))

            # ee_vel
            fig.add_trace(
                go.Scatter(
                    y=ee_vel[:-1, 0],
                    mode="lines+markers",
                    name="ee_x",
                    line=dict(color="red", dash="dashdot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=ee_vel[:-1, 1],
                    mode="lines+markers",
                    name="ee_z",
                    line=dict(color="blue", dash="dashdot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=ee_vel[:-1, 2],
                    mode="lines+markers",
                    name="ee_rot",
                    line=dict(color="green", dash="dashdot"),
                )
            )

            plot_name = "ee_vel_comp"
            fig.update_layout(
                title=f"Episode {self.ep_num} - EE Vel Comp",
                xaxis_title="Time step",
                yaxis_title=y_label,
            )
            save_path = self.plots_dir / f"ep_{self.ep_num}" / f"{plot_name}_{env_idx}.html"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path, auto_play=False)

    def _process_eval_vars(self, env_idx) -> None:
        for var_name, eval_types in self.vars_to_eval.items():
            for each_eval_type in eval_types:
                if each_eval_type == "sum":
                    proc_var = np.sum(np.abs(self.eval_vars_values[var_name][env_idx]))
                elif each_eval_type == "mean":
                    proc_var = np.mean(self.eval_vars_values[var_name][env_idx])
                elif each_eval_type == "rms":
                    proc_var = np.sqrt(np.mean(np.square(self.eval_vars_values[var_name][env_idx])))
                elif each_eval_type == "max":
                    proc_var = np.max(self.eval_vars_values[var_name][env_idx])
                elif each_eval_type == "last":
                    proc_var = self.eval_vars_values[var_name][env_idx][-1]
                else:
                    raise ValueError(
                        f"Invalid eval type ({each_eval_type}) for var: {var_name}. Must be 'sum', 'mean', 'rms', or 'max'."
                    )

                self.eval_vars_proc_values[var_name][each_eval_type].append(proc_var)

            self.eval_vars_values[var_name][env_idx] = []

    def print_results(self):
        print("\n---- Mean Processed values ----")
        for each_eval_var, eval_types in self.vars_to_eval.items():
            print(f"\n---- {each_eval_var} ----")
            # if each_eval_var == "socket_x":
            #     # print(f"\t-{self.eval_vars_proc_values[each_eval_var]['last']}")
            #     continue

            for each_eval_type in eval_types:
                proc_var = np.mean(self.eval_vars_proc_values[each_eval_var][each_eval_type])
                print(f"\t-{each_eval_type}: {proc_var:.4f}")
