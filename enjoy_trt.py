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
import pprint
import subprocess
import io
import traceback
import textwrap
import re
import logging
logger = logging.getLogger(__name__)

from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import MessageToDict

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorflow as tf

# orig_tf_cast = tf.cast
# def wrap_tf_cast(*args, **kwargs):
#     # import ipdb; ipdb.set_trace()
#     buf = io.StringIO()
#     traceback.print_stack(file=buf)
#     print("> tf.cast(...) called:\n{msg}".format(msg=textwrap.indent(buf.getvalue(), prefix='  ')))
#     return orig_tf_cast(*args, **kwargs)
# tf.cast = wrap_tf_cast

# orig_tf_random_uniform = tf.random.uniform
# def wrap_tf_random_uniform(*args, **kwargs):
#     # import ipdb; ipdb.set_trace()
#     buf = io.StringIO()
#     traceback.print_stack(file=buf)
#     print("> tf.random.uniform(...) called:\n{msg}".format(msg=textwrap.indent(buf.getvalue(), prefix='  ')))
#     return orig_tf_random_uniform(*args, **kwargs)
# tf.random.uniform = wrap_tf_random_uniform

# orig_tf_strided_slice = tf.strided_slice
# def wrap_tf_strided_slice(*args, **kwargs):
#     # import ipdb; ipdb.set_trace()
#     buf = io.StringIO()
#     traceback.print_stack(file=buf)
#     print("> tf.strided_slice(...) called:\n{msg}".format(msg=textwrap.indent(buf.getvalue(), prefix='  ')))
#     return orig_tf_strided_slice(*args, **kwargs)
# tf.strided_slice = wrap_tf_strided_slice

import tensorrt as trt

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
    def __init__(self, path, mode='r', empty_js=None, debug=False):
        """
        :param path:
        :param mode:
            r  => read the json file (it should exist)
            w  => DON'T read json file (even if it exists)
            rw => read the json file if it exists
        :param empty_js:
            Template js object to use if no json file exists.
        :param debug:
        """
        self.path = path
        self.mode = mode
        assert mode in {'r', 'w', 'rw'}
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
        if self.mode == 'r' and not _e(self.path):
            raise RuntimeError("Couldn't read json file from {path} since it didn't exist".format(path=self.path))

        if 'r' in self.mode and _e(self.path):
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
    setup_logging()
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
    parser.add_argument('--debug', action='store_true',
                        help='debug')


    parser.add_argument('--mode', help='IML: mode of execution',
                        choices=[
                            'default',
                            'microbench_iml_python_annotation',
                            'microbench_iml_clib_interception_simulator',
                            'microbench_iml_clib_interception_tensorflow',
                            'save_tensorrt',
                            'load_tensorrt',
                            'microbench_inference',
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
    args, argv = parser.parse_known_args()

    if args.mode == 'microbench_inference':
        subparser = argparse.ArgumentParser("Microbenchmark TensorFlow inference throughput/latency on random data.")
        subparser.add_argument('--batch-size', type=int, default=1, help="Number of random samples per minibatch")
        subparser.add_argument('--inference-starts', type=int, default=0)
        subparser.add_argument('--warmup-iters', type=int, default=100)
        subparser_args = subparser.parse_args(argv)
        for attr, value in vars(subparser_args).items():
            if hasattr(args, attr):
                raise RuntimeError("Main argument parser already has --{opt}, but {mode} subparser has conflicting --{opt}".format(
                    opt=attr,
                    mode=args.mode,
                ))
                # assert not hasattr(args, attr)
            setattr(args, attr, value)

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

    def build_engine(self, uff_model_file, input_names, input_shapes, output_names, workspace_size_bytes=None):
        if workspace_size_bytes is None:
            workspace_size_bytes = GiB(1)
        network_flags = 0
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_flags) as network, trt.UffParser() as parser:
            builder.max_workspace_size = workspace_size_bytes
            # Parse the Uff Network
            def get_node_name(node_name):
                m = re.search(r'(?P<node>.*):\d+$', node_name)
                if m:
                    return m.group('node')
                return node_name
            for input_name, input_shape in zip(input_names, input_shapes):

                # NOTE: trtexec does NOT include a batch dimension (i.e., batch is first dimension and it's "implicit")
                new_input_shape = []
                for i, dim in enumerate(input_shape):
                    if dim == -1:
                        assert i == 0
                        continue
                    else:
                        new_input_shape.append(dim)

                new_input_name = get_node_name(input_name)
                logger.info(trt_log_msg(f"parser.register_input: {new_input_name}, {new_input_shape}"))
                # NOTE: default format is trt.UffInputOrder.NCHW, but [?, 84, 84, 4] is NHWC.
                parser.register_input(new_input_name, new_input_shape, trt.tensorrt.UffInputOrder.NHWC)
            for output_name in output_names:
                new_output_name = get_node_name(output_name)
                logger.info(trt_log_msg(f"parser.register_output: {new_output_name}"))
                parser.register_output(new_output_name)
            parse_ret = parser.parse(uff_model_file, network)
            if not parse_ret:
                raise RuntimeError("tensorrt.UffParser failed to parse model @ {path}".format(path=uff_model_file))

            # Build and return an engine.
            engine = builder.build_cuda_engine(network)
            assert engine is not None
            return engine

    def uff_model_path(self, model_base):
        return "{base}.uff".format(
            base=model_base)

    def tf_model_path(self, model_base):
        return "{base}.pb".format(
            base=model_base)

    def tf_model_json_path(self, model_base):
        return "{base}.metadata.json".format(
            base=model_base)

    def tf_model_inputs_path(self, model_base):
        return "{base}.inputs.txt".format(
            base=model_base)

    def tf_model_outputs_path(self, model_base):
        return "{base}.outputs.txt".format(
            base=model_base)

    def with_trt_model(self, model_base, func):
        # data_paths, _ = common.find_sample_data(description="Runs an MNIST network using a UFF model file", subfolder="mnist")
        # model_path = os.environ.get("MODEL_PATH") or os.path.join(os.path.dirname(__file__), "models")
        # model_file = os.path.join(model_path, ModelData.MODEL_FILE)

        js = JsonFile(self.tf_model_json_path(model_base), mode='r')
        input_names = [node["name"] for node in js["inputs"]]
        input_shapes = [node["shape"] for node in js["inputs"]]
        output_names = [node["name"] for node in js["outputs"]]
        with self.build_engine(self.uff_model_path(model_base), input_names, input_shapes, output_names) as engine:
            # Build an engine, allocate buffers and create a stream.
            # For more information on buffer allocation, refer to the introductory samples.
            # inputs, outputs, bindings, stream = allocate_buffers(engine)
            with engine.create_execution_context() as execution_context:
                # For more information on performing inference, refer to the introductory samples.
                # The common.do_inference function will return a list of outputs - we only have one in this case.
                trt_ctx = TRTContext(engine, execution_context)
                ret = func(trt_ctx)
                return ret

    def save_tf_model(self, model, model_base):
        # First freeze the graph and remove training nodes.
        # output_names = model.output.op.name
        inputs = model.inputs()
        outputs = model.outputs()
        # NOTE: node.name has a trailing ":0", presumably tell us the device it's been placed on.
        output_names = [node.op.name for node in outputs]
        # sess = tf.keras.backend.get_session()
        # sess = tf.compat.v1.get_default_session()
        sess = model.sess
        graph = model.graph
        # graph = outputs[0].graph
        # graph = tf.compat.v1.get_default_graph()
        # graph = sess.graph
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(),
            # [output_names]
            # outputs,
            output_names,
        )
        frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
        # Save the model
        print(f"> Save TensorFlow proto model to {model_base}")

        js = JsonFile(self.tf_model_json_path(model_base), mode='w')

        def _output_nodes(js, nodes):
            for node in nodes:
                shape = []
                for dim in node.shape.as_list():
                    if dim is None:
                        shape.append(-1)
                    else:
                        shape.append(dim)
                js.append({
                    'name': node.name,
                    'shape': shape,
                })
        js['outputs'] = []
        _output_nodes(js['outputs'], outputs)
        js['inputs'] = []
        _output_nodes(js['inputs'], inputs)

        def _output_all_nodes(js, nodes):
            def _as_dict(proto):
                return dict((key, value) for key, value in proto.items())

            def _remove_keys(dic, keys):
                for k in keys:
                    if k in dic:
                        del dic[k]

            for node in nodes:
                node_js = MessageToDict(node)
                if ( "attr" in node_js
                        and "value" in node_js["attr"]
                        and "tensor" in node_js["attr"]["value"]
                        and "tensorContent" in node_js["attr"]["value"]["tensor"]
                    ):
                    del node_js["attr"]["value"]["tensor"]["tensorContent"]

                js.append(node_js)

        js['nodes'] = []
        _output_all_nodes(js['nodes'], frozen_graph.node)

        js.dump()

        with open(self.tf_model_path(model_base), "wb") as ofile:
            ofile.write(frozen_graph.SerializeToString())

        # Needed for tf2onnx tool when converting from "graphdef" format
        # https://github.com/onnx/tensorflow-onnx#getting-started

        with open(self.tf_model_outputs_path(model_base), "w") as f:
            for node in outputs:
                f.write(node.name)
                f.write("\n")

        with open(self.tf_model_inputs_path(model_base), "w") as f:
            for node in inputs:
                f.write(node.name)
                f.write("\n")

    def mode_save_tensorrt(self):
        args = self.args
        parser = self.parser

        algo = args.algo

        # self.handle_iml(reports_progress=False)
        env = self.make_env()
        model = self.make_model(env)
        env_id = args.env

        is_atari = self.is_atari()

        obs = env.reset()

        model_base = _j(".", "tf_model")
        tf_model_path = self.tf_model_path(model_base)
        uff_model_path = self.uff_model_path(model_base)
        self.save_tf_model(model, model_base)
        subprocess.check_call(['convert-to-uff', tf_model_path])

    def mode_load_tensorrt(self):
        args = self.args
        parser = self.parser

        algo = args.algo

        # self.handle_iml(reports_progress=False)
        env = self.make_env()
        model = self.make_model(env)
        env_id = args.env

        is_atari = self.is_atari()

        obs = env.reset()

        model_base = _j(".", "tf_model")
        tf_model_path = self.tf_model_path(model_base)
        uff_model_path = self.uff_model_path(model_base)

        # Make sure we can run inference using the model on TensorRT
        if os.path.exists(uff_model_path):
            print("> Running inference with TensorRT...")
            def inference(trt_ctx):
                obs = env.observation_space.low
                ret = trt_ctx.inference([obs])
                print("> Ran inference with TensorRT:\n{msg}".format(msg=pprint.pformat({
                    'obs': obs,
                    'ret': ret,
                })))
            self.with_trt_model(model_base, inference)

    def mode_microbench_inference(self):
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
        deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her'] and not args.stochastic

        obs_space = env.observation_space
        # Q: Does observation_space already include the batch size...?

        def random_minibatch(batch_size, obs_space):
            shape = (batch_size,) + obs_space.shape
            batch = np.random.uniform(size=shape)
            return batch

        def is_warmed_up(t, operations_seen, operations_available):
            """
            Return true once we are executing the full training-loop.

            :return:
            """
            assert operations_seen.issubset(operations_available)
            # can_sample = self.replay_buffer.can_sample(self.batch_size)
            # return can_sample and operations_seen == operations_available and self.num_timesteps > self.learning_starts

            return operations_seen == operations_available and \
                   t > args.warmup_iters and \
                   ( args.inference_starts == 0 or t > args.inference_starts )

        operations_available = {'inference_loop', 'inference'}
        operations_seen = set()
        def iml_prof_operation(operation):
            should_skip = operation not in operations_available
            op = iml.prof.operation(operation, skip=should_skip)
            if not should_skip:
                operations_seen.add(operation)
            return op

        process_name = get_process_name(args)
        phase_name = process_name
        random_obs = random_minibatch(args.batch_size, obs_space)
        logger.info(log_msg('INFER', "using --batch-size={batch} => {shape}".format(
            batch=args.batch_size,
            shape=random_obs.shape,
        )))
        # Q: How long to run for...?
        # Shouldn't matter...? Long enough to obtain SM metrics.
        # Q: Will it end early?
        # Q: Can we do multiple minibatch sizes in a single run?

        # Set some default trace-collection termination conditions (if not set via the cmdline).
        # These were set via experimentation until training ran for "sufficiently long" (e.g. 2-4 minutes).
        #
        # NOTE: DQN and SAC both call iml.prof.report_progress after each timestep
        # (hence, we run lots more iterations than DDPG/PPO).
        #iml.prof.set_max_training_loop_iters(10000, skip_if_set=True)
        #iml.prof.set_delay_training_loop_iters(10, skip_if_set=True)
        # iml.prof.set_max_passes(10, skip_if_set=True)
        # 1 configuration pass.
        # iml.prof.set_delay_passes(1, skip_if_set=True)
        iml.prof.set_max_training_loop_iters(args.n_timesteps*2, skip_if_set=False)

        if not iml.prof.delay:
            parser.error("You must use --iml-delay for --mode={mode}".format(mode=args.mode))

        with iml.prof.profile(process_name=process_name, phase_name=phase_name):
            # episode_reward = 0.0
            # episode_rewards = []
            # ep_len = 0
            # # For HER, monitor success rate
            # successes = []

            warmed_up = False
            inference_time_sec = []
            for t in range(args.n_timesteps):

                if iml.prof.delay and is_warmed_up(t, operations_seen, operations_available) and not iml.prof.tracing_enabled:
                    # Entire training loop is now running; enable IML tracing
                    logger.info(log_msg('RLS', "ENABLE TRACING"))
                    iml.prof.enable_tracing()
                    start_t = time.time()
                    start_timesteps = t
                    warmed_up = True

                if args.debug:
                    logger.info(
                        log_msg('RLS',
                                textwrap.dedent(f"""\
                                @ t={t}: operations_seen = {operations_seen}
                                  waiting for = {operations_available.difference(operations_seen)}
                    """)).rstrip())
                if operations_seen == operations_available:
                    operations_seen.clear()
                    if args.debug:
                        logger.info(log_msg('RLS', f"iml.prof.report_progress: t={t}"))
                    iml.prof.report_progress(
                        percent_complete=t/float(args.n_timesteps),
                        num_timesteps=t,
                        total_timesteps=args.n_timesteps)

                with iml_prof_operation('inference_loop'):
                    with iml_prof_operation('inference'):
                        start_inference_t = time.time()
                        action, _ = model.predict(random_obs, deterministic=deterministic)
                        end_inference_t = time.time()
                        if warmed_up:
                            inf_time_sec = end_inference_t - start_inference_t
                            inference_time_sec.append(inf_time_sec)

                        # Random Agent
                        # action = [env.action_space.sample()]
                        # Clip Action to avoid out of bound errors
                        # if isinstance(env.action_space, gym.spaces.Box):
                        #     action = np.clip(action, env.action_space.low, env.action_space.high)
                    # with iml.prof.operation('step'):
                    #     obs, reward, done, infos = env.step(action)
                    # if not args.no_render:
                    #     env.render('human')
                    #
                    # episode_reward += reward[0]
                    # ep_len += 1
                    #
                    # if args.n_envs == 1:
                    #     # For atari the return reward is not the atari score
                    #     # so we have to get it from the infos dict
                    #     if is_atari and infos is not None and args.verbose >= 1:
                    #         episode_infos = infos[0].get('episode')
                    #         if episode_infos is not None:
                    #             print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                    #             print("Atari Episode Length", episode_infos['l'])
                    #
                    #     if done and not is_atari and args.verbose > 0:
                    #         # NOTE: for env using VecNormalize, the mean reward
                    #         # is a normalized reward when `--norm_reward` flag is passed
                    #         print("Episode Reward: {:.2f}".format(episode_reward))
                    #         print("Episode Length", ep_len)
                    #         episode_rewards.append(episode_reward)
                    #         episode_reward = 0.0
                    #         ep_len = 0
                    #
                    #     # Reset also when the goal is achieved when using HER
                    #     if done or infos[0].get('is_success', False):
                    #         if args.algo == 'her' and args.verbose > 1:
                    #             print("Success?", infos[0].get('is_success', False))
                    #         # Alternatively, you can add a check to wait for the end of the episode
                    #         # if done:
                    #         obs = env.reset()
                    #         if args.algo == 'her':
                    #             successes.append(infos[0].get('is_success', False))
                    #             episode_reward, ep_len = 0.0, 0

            end_t = time.time()
            iml.prof.report_progress(
                percent_complete=1,
                num_timesteps=args.n_timesteps,
                total_timesteps=args.n_timesteps)

            inference_time_sec = np.array(inference_time_sec)
            # total_time_sec = (end_t - start_t)
            total_time_sec = np.sum(inference_time_sec)
            # total_samples = (args.n_timesteps - start_timesteps)*args.batch_size
            total_samples = len(inference_time_sec)*args.batch_size
            throughput_qps = total_samples / total_time_sec
            inference_js = dict()
            inference_js['raw_samples'] = dict()
            inference_js['summary_metrics'] = dict()
            inference_js['raw_samples']['inference_time_sec'] = inference_time_sec.tolist()
            inference_js['summary_metrics']['throughput_qps'] = throughput_qps
            inference_js['summary_metrics']['total_samples'] = total_samples
            inference_js['summary_metrics']['batch_size'] = args.batch_size
            inference_js['summary_metrics']['total_time_sec'] = total_time_sec
            inference_js['summary_metrics']['mean_inference_time_sec'] = inference_time_sec.mean()
            inference_js['summary_metrics']['inference_time_percentile_99_sec'] = np.percentile(inference_time_sec, 0.99)
            do_dump_json(inference_js, _j(iml.prof.directory, 'mode_microbench_inference.json'))


            # if args.verbose > 0 and len(successes) > 0:
            #     print("Success rate: {:.2f}%".format(100 * np.mean(successes)))
            #
            # if args.verbose > 0 and len(episode_rewards) > 0:
            #     print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))

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
        elif args.mode == 'save_tensorrt':
            self.mode_save_tensorrt()
        elif args.mode == 'load_tensorrt':
            self.mode_load_tensorrt()
        elif args.mode == 'microbench_inference':
            self.mode_microbench_inference()
        else:
            raise NotImplementedError("Note sure how to run --mode={mode}".format(
                mode=args.mode))


# You can set the logger severity higher to suppress messages (or lower to display more messages).
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class ModelData(object):
    MODEL_FILE = "lenet5.uff"
    INPUT_NAME ="input_1"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "dense_1/Softmax"

def GiB(val):
    return val * 1 << 30

def build_engine(model_file):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = GiB(1)
        # Parse the Uff Network
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_file, network)
        # Build and return an engine.
        engine = builder.build_cuda_engine(network)
        assert engine is not None
        return engine

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def size_bytes(self):
        return self.host.nbytes

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        max_batch_size = engine.max_batch_size
        shape = engine.get_binding_shape(binding)
        volume = trt.volume(shape)
        size = volume * max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        assert np.dtype(dtype).itemsize*size == host_mem.nbytes
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            buffer_type = 'input'
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            buffer_type = 'output'
            outputs.append(HostDeviceMem(host_mem, device_mem))
        logger.info(trt_log_msg(f"buftype={buffer_type}, binding={binding}, shape={shape}, max_batch_size={max_batch_size}, volume={volume}, dtype={dtype}"))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class TRTContext:
    def __init__(self,
                 engine,
                 execution_context,
                 # execution_context, bindings, inputs, outputs, stream
                 ):
        self.engine = engine
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.execution_context = execution_context

    def inference(self, inputs):
        assert len(self.inputs) == len(inputs)
        for i in range(len(inputs)):
            # assert inputs[i].nbytes == self.inputs[i].size_bytes()
            # NOTE: np.copyto will allow conversion from uint8 to float32
            assert inputs[i].size == self.inputs[i].size

        for i in range(len(inputs)):
            # Copy to page-locked host-side buffer.
            # do_inference will copy from self.inputs[i].host to self.inputs[i].device
            np.copyto(self.inputs[i].host, np.ravel(inputs[i]))

        outputs = do_inference(self.execution_context, self.bindings, self.inputs, self.outputs, self.stream)
        assert len(outputs) == 1
        output = outputs[0]

        return output

def do_dump_json(data, path):
    os.makedirs(_d(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data,
                  f,
                  sort_keys=True,
                  indent=4,
                  skipkeys=False)

def log_msg(tag, msg):
    return f"[{tag}] {msg}"

def trt_log_msg(msg):
    return log_msg('TRT', msg)

def setup_logging():
    format = 'PID={process}/{processName} @ {funcName}, {filename}:{lineno} :: {asctime} {levelname}: {message}'
    logging.basicConfig(format=format, style='{')
    logger.setLevel(logging.INFO)

if __name__ == '__main__':
    main()
