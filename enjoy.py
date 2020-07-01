import argparse
import os
import warnings
import copy
import time
import sys
import pkg_resources
import importlib
import json
import codecs

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

# For pybullet envs
warnings.filterwarnings("ignore")
import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
try:
    import highway_env
except ImportError:
    highway_env = None
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

from stable_baselines.iml import wrap_pybullet, unwrap_pybullet

import iml_profiler.api as iml

# Fix for breaking change in v2.6.0
if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
    sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
    stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

def load_json(path):
    with codecs.open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def dump_json(data, path):
    os.makedirs(_d(path), exist_ok=True)
    json.dump(data,
              codecs.open(path, mode='w', encoding='utf-8'),
              sort_keys=True, indent=4,
              skipkeys=False)

class JsonFile:
    def __init__(self, path, empty_js=None, debug=False):
        """
        :param path:
        :param empty_js:
            Template js object to use if no json file exists.
        :param debug:
        """
        self.path = path
        self.empty_js = None
        self.debug = debug
        self._load()

    def append(self, key, value):
        if key not in self.js:
            self.js[key] = []
        self.js[key].append(value)

    def dump(self):
        if self.debug:
            print("> Dump json @ {path}".format(
                path=self.path,
            ))
        dump_json(self.js, self.path)

    def _load(self):
        if _e(self.path):
            if self.debug:
                print("> Load json @ {path}".format(
                    path=self.path,
                ))
            self.js = load_json(self.path)
            return

        if self.empty_js is not None:
            self.js = copy.copy(self.empty_js)
        else:
            self.js = dict()

    def get(self, key, dflt=None):
        return self.js.get(key, dflt)

    def __setitem__(self, key, value):
        self.js[key] = value

    def __getitem__(self, key):
        return self.js[key]

    def __len__(self):
        return len(self.js)

def get_process_name(args):
    process_name = "{algo}_{env}".format(
        algo=args.algo,
        env=args.env)
    return process_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                        type=int)
    parser.add_argument('--n-envs', help='number of environments', default=1,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=-1,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)


    parser.add_argument('--mode', help='IML: mode of execution',
                        choices=[
                            'default',
                            'microbench_iml_python_annotation',
                            'microbench_iml_clib_interception_simulator',
                            'microbench_iml_clib_interception_tensorflow',
                        ],
                        default='default')
    parser.add_argument('--iterations', help='microbenchmark mode: iterations', type=int, default=1000)
    # Run repetitions from external script.
    parser.add_argument('--repetition', help='microbenchmark mode: repetitions', type=int)

    parser.add_argument('--no-render', action='store_true', default=False,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    iml.add_iml_arguments(parser)
    iml.register_wrap_module(wrap_pybullet, unwrap_pybullet)
    args = parser.parse_args()

    trained_agent = TrainedAgent(args, parser)
    trained_agent.run()

class TrainedAgent:
    def __init__(self, args, parser):
        self.args = args
        self.parser = parser

    def is_atari(self):
        args = self.args
        parser = self.parser
        env_id = args.env
        is_atari = 'NoFrameskip' in env_id
        return is_atari

    def get_iml_directory(self):
        # Use the --reward-log argument to store traces (if provided).
        # Otherwise, require the user to provide --iml-directory.
        args = self.args
        log_dir = self.log_dir()
        if self.is_microbench():
            if args.repetition is None:
                raise RuntimeError("ERROR: you must provide --repetition")
            if args.iml_directory is None:
                raise RuntimeError("ERROR: you must provide --iml-directory")
            # Add "repetition_\d+" sub-folder.
            iml_directory = "{iml_dir}/{mode}/repetition_{r:02d}".format(
                iml_dir=args.iml_directory,
                mode=args.mode,
                r=args.repetition,
            )
        elif log_dir is not None:
            # <--reward-log>/iml_traces
            iml_directory = os.path.join(log_dir, 'iml_traces')
        else:
            # User must provide --iml-directory
            iml_directory = args.iml_directory
        return iml_directory

    def mode_json_path(self):
        assert self.is_microbench()
        args = self.args

        if args.iml_directory is None:
            raise RuntimeError("ERROR: you must provide --iml-directory")
        path = "{iml_dir}/{mode}/{mode}.json".format(
            iml_dir=args.iml_directory,
            mode=args.mode,
            r=args.repetition,
        )

        return path

    def log_dir(self):
        args = self.args
        log_dir = args.reward_log if args.reward_log != '' else None
        return log_dir

    def is_microbench(self):
        args = self.args
        return args.mode in {
            'microbench_iml_python_annotation',
            'microbench_iml_clib_interception_simulator',
            'microbench_iml_clib_interception_tensorflow',
        }

    def handle_iml(self, reports_progress):
        args = self.args
        parser = self.parser
        iml_directory = self.get_iml_directory()
        iml.handle_iml_args(parser, args, directory=iml_directory, reports_progress=reports_progress)
        iml.prof.set_metadata({
            'algo': args.algo,
            'env': args.env,
        })

    def make_env(self):
        args = self.args
        parser = self.parser

        log_dir = self.log_dir()

        # Going through custom gym packages to let them register in the global registory
        for env_module in args.gym_packages:
            importlib.import_module(env_module)

        env_id = args.env
        algo = args.algo

        log_path = self.log_path()


        if algo in ['dqn', 'ddpg', 'sac']:
            args.n_envs = 1

        set_global_seeds(args.seed)

        is_atari = self.is_atari()

        stats_path = os.path.join(log_path, env_id)
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

        env = create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
                              stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                              should_render=not args.no_render,
                              hyperparams=hyperparams)
        return env

    def log_path(self):
        args = self.args
        parser = self.parser

        env_id = args.env
        algo = args.algo
        folder = args.folder

        if args.exp_id == 0:
            args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
            print('Loading latest experiment, id={}'.format(args.exp_id))

        # Sanity checks
        if args.exp_id > 0:
            log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
        else:
            log_path = os.path.join(folder, algo)

        return log_path


    def make_model(self, env):
        args = self.args
        parser = self.parser

        log_path = self.log_path()

        algo = args.algo
        env_id = args.env

        model_path = "{}/{}.pkl".format(log_path, env_id)

        assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
        assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, env_id, model_path)

        # ACER raises errors because the environment passed must have
        # the same number of environments as the model was trained on.
        load_env = None if algo == 'acer' else env
        model = ALGOS[algo].load(model_path, env=load_env)
        return model

    def mode_default(self):
        args = self.args
        parser = self.parser

        algo = args.algo

        self.handle_iml(reports_progress=False)
        env = self.make_env()
        model = self.make_model(env)
        env_id = args.env

        is_atari = self.is_atari()

        obs = env.reset()

        # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
        deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her'] and not args.stochastic

        process_name = get_process_name(args)
        phase_name = process_name
        with iml.prof.profile(process_name=process_name, phase_name=phase_name):
            episode_reward = 0.0
            episode_rewards = []
            ep_len = 0
            # For HER, monitor success rate
            successes = []
            with iml.prof.operation('inference_loop'):
                for _ in range(args.n_timesteps):
                    with iml.prof.operation('inference'):
                        action, _ = model.predict(obs, deterministic=deterministic)
                        # Random Agent
                        # action = [env.action_space.sample()]
                        # Clip Action to avoid out of bound errors
                        if isinstance(env.action_space, gym.spaces.Box):
                            action = np.clip(action, env.action_space.low, env.action_space.high)
                    with iml.prof.operation('step'):
                        obs, reward, done, infos = env.step(action)
                    if not args.no_render:
                        env.render('human')

                    episode_reward += reward[0]
                    ep_len += 1

                    if args.n_envs == 1:
                        # For atari the return reward is not the atari score
                        # so we have to get it from the infos dict
                        if is_atari and infos is not None and args.verbose >= 1:
                            episode_infos = infos[0].get('episode')
                            if episode_infos is not None:
                                print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                                print("Atari Episode Length", episode_infos['l'])

                        if done and not is_atari and args.verbose > 0:
                            # NOTE: for env using VecNormalize, the mean reward
                            # is a normalized reward when `--norm_reward` flag is passed
                            print("Episode Reward: {:.2f}".format(episode_reward))
                            print("Episode Length", ep_len)
                            episode_rewards.append(episode_reward)
                            episode_reward = 0.0
                            ep_len = 0

                        # Reset also when the goal is achieved when using HER
                        if done or infos[0].get('is_success', False):
                            if args.algo == 'her' and args.verbose > 1:
                                print("Success?", infos[0].get('is_success', False))
                            # Alternatively, you can add a check to wait for the end of the episode
                            # if done:
                            obs = env.reset()
                            if args.algo == 'her':
                                successes.append(infos[0].get('is_success', False))
                                episode_reward, ep_len = 0.0, 0

            if args.verbose > 0 and len(successes) > 0:
                print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

            if args.verbose > 0 and len(episode_rewards) > 0:
                print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))

            # Workaround for https://github.com/openai/gym/issues/893
            if not args.no_render:
                if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
                    # DummyVecEnv
                    # Unwrap env
                    while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                        env = env.venv
                    env.envs[0].env.close()
                else:
                    # SubprocVecEnv
                    env.close()

    def deterministic(self):
        args = self.args
        parser = self.parser
        algo = args.algo
        # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
        deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her'] and not args.stochastic
        return deterministic

    def mode_microbench_iml_python_annotation(self):
        args = self.args
        parser = self.parser

        self.handle_iml(reports_progress=True)

        process_name = get_process_name(args)
        phase_name = process_name
        with iml.prof.profile(process_name=process_name, phase_name=phase_name):

            if iml.prof.delay and not iml.prof.tracing_enabled:
                # Entire training loop is now running; enable IML tracing
                iml.prof.enable_tracing()

            iml.prof.report_progress(
                percent_complete=0/float(args.iterations),
                num_timesteps=0,
                total_timesteps=args.iterations)

            iterations_start_t = time.time()
            for i in range(args.iterations):
                with iml.prof.operation('iteration'):
                    pass
            iterations_end_t = time.time()

            iml.prof.report_progress(
                percent_complete=1.,
                num_timesteps=args.iterations,
                total_timesteps=args.iterations)

            iterations_total_sec = iterations_end_t - iterations_start_t
            time_sec_per_iteration = iterations_total_sec / args.iterations

            json_path = self.mode_json_path()
            js = JsonFile(json_path)
            js.append('iterations_total_sec', iterations_total_sec)
            js.append('iterations', args.iterations)
            js.append('time_sec_per_iteration', time_sec_per_iteration)
            js['repetitions'] = js.get('repetitions', 0) + 1
            print("> Dump --mode={mode} results @ {path}".format(
                path=json_path,
                mode=args.mode,
            ))
            js.dump()

    def mode_microbench_iml_clib_interception_simulator(self):
        args = self.args
        parser = self.parser

        algo = args.algo

        self.handle_iml(reports_progress=True)
        env = self.make_env()
        # model = self.make_model()
        env_id = args.env

        is_atari = self.is_atari()

        obs = env.reset()

        # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
        deterministic = self.deterministic()

        process_name = get_process_name(args)
        phase_name = process_name
        with iml.prof.profile(process_name=process_name, phase_name=phase_name):

            if iml.prof.delay and not iml.prof.tracing_enabled:
                # Entire training loop is now running; enable IML tracing
                iml.prof.enable_tracing()

            iml.prof.report_progress(
                percent_complete=0/float(args.iterations),
                num_timesteps=0,
                total_timesteps=args.iterations)

            # episode_reward = 0.0
            # episode_rewards = []
            # ep_len = 0
            # For HER, monitor success rate
            # successes = []
            iterations_start_t = time.time()
            for i in range(args.iterations):
                with iml.prof.operation('iteration'):
                    # action, _ = model.predict(obs, deterministic=deterministic)
                    # Random Agent
                    action = [env.action_space.sample()]
                    # Clip Action to avoid out of bound errors
                    if isinstance(env.action_space, gym.spaces.Box):
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, reward, done, infos = env.step(action)

                    # episode_reward += reward[0]
                    # ep_len += 1
            iterations_end_t = time.time()

            iml.prof.report_progress(
                percent_complete=1.,
                num_timesteps=args.iterations,
                total_timesteps=args.iterations)

            iterations_total_sec = iterations_end_t - iterations_start_t
            time_sec_per_iteration = iterations_total_sec / args.iterations

            json_path = self.mode_json_path()
            js = JsonFile(json_path)
            js.append('iterations_total_sec', iterations_total_sec)
            js.append('iterations', args.iterations)
            js.append('time_sec_per_iteration', time_sec_per_iteration)
            js['repetitions'] = js.get('repetitions', 0) + 1
            print("> Dump --mode={mode} results @ {path}".format(
                path=json_path,
                mode=args.mode,
            ))
            js.dump()

            # Workaround for https://github.com/openai/gym/issues/893
            if not args.no_render:
                if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
                    # DummyVecEnv
                    # Unwrap env
                    while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                        env = env.venv
                    env.envs[0].env.close()
                else:
                    # SubprocVecEnv
                    env.close()

    def mode_microbench_iml_clib_interception_tensorflow(self):
        args = self.args
        parser = self.parser

        algo = args.algo

        self.handle_iml(reports_progress=True)
        env = self.make_env()
        model = self.make_model(env)
        env_id = args.env

        is_atari = self.is_atari()

        obs = env.reset()

        # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
        deterministic = self.deterministic()

        process_name = get_process_name(args)
        phase_name = process_name
        with iml.prof.profile(process_name=process_name, phase_name=phase_name):

            if iml.prof.delay and not iml.prof.tracing_enabled:
                # Entire training loop is now running; enable IML tracing
                iml.prof.enable_tracing()

            iml.prof.report_progress(
                percent_complete=0/float(args.iterations),
                num_timesteps=0,
                total_timesteps=args.iterations)

            # episode_reward = 0.0
            # episode_rewards = []
            # ep_len = 0
            # For HER, monitor success rate
            # successes = []
            iterations_start_t = time.time()
            for i in range(args.iterations):
                with iml.prof.operation('iteration'):
                    action, _ = model.predict(obs, deterministic=deterministic)
                    # Random Agent
                    # action = [env.action_space.sample()]
                    # Clip Action to avoid out of bound errors
                    if isinstance(env.action_space, gym.spaces.Box):
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                    # obs, reward, done, infos = env.step(action)

                    # episode_reward += reward[0]
                    # ep_len += 1
            iterations_end_t = time.time()

            iml.prof.report_progress(
                percent_complete=1.,
                num_timesteps=args.iterations,
                total_timesteps=args.iterations)

            iterations_total_sec = iterations_end_t - iterations_start_t
            time_sec_per_iteration = iterations_total_sec / args.iterations

            json_path = self.mode_json_path()
            js = JsonFile(json_path)
            js.append('iterations_total_sec', iterations_total_sec)
            js.append('iterations', args.iterations)
            js.append('time_sec_per_iteration', time_sec_per_iteration)
            js['repetitions'] = js.get('repetitions', 0) + 1
            print("> Dump --mode={mode} results @ {path}".format(
                path=json_path,
                mode=args.mode,
            ))
            js.dump()

            # Workaround for https://github.com/openai/gym/issues/893
            if not args.no_render:
                if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
                    # DummyVecEnv
                    # Unwrap env
                    while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                        env = env.venv
                    env.envs[0].env.close()
                else:
                    # SubprocVecEnv
                    env.close()

    def run(self):
        args = self.args
        parser = self.parser

        if args.mode == 'default':
            self.mode_default()
        elif args.mode == 'microbench_iml_python_annotation':
            self.mode_microbench_iml_python_annotation()
        elif args.mode == 'microbench_iml_clib_interception_simulator':
            self.mode_microbench_iml_clib_interception_simulator()
        elif args.mode == 'microbench_iml_clib_interception_tensorflow':
            self.mode_microbench_iml_clib_interception_tensorflow()
        else:
            raise NotImplementedError("Note sure how to run --mode={mode}".format(
                mode=args.mode))


if __name__ == '__main__':
    main()
