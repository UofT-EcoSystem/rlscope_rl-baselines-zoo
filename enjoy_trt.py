import logging
logger = logging.getLogger(__name__)
import textwrap
import pprint


import argparse
import os
import itertools
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
import multiprocessing
import threading
import numpy as np
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent import futures

# from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import MessageToDict

# import pycuda.driver as cuda
# # This import causes pycuda to automatically manage CUDA context creation and cleanup.
# import pycuda.autoinit


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

# import tensorrt as trt

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

# For pybullet envs
warnings.filterwarnings("ignore")
# pybullet_envs = None
# highway_env = None
# gym = None
# import gym
# try:
#     import pybullet_envs
# except ImportError:
#     pybullet_envs = None
# try:
#     import highway_env
# except ImportError:
#     highway_env = None

# import stable_baselines
# from stable_baselines.common import set_global_seeds
# from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv


# import stable_baselines
# from stable_baselines.common import set_global_seeds
# from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv
# from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
# from stable_baselines.iml import wrap_pybullet, unwrap_pybullet
# import iml_profiler.api as iml

# import tensorflow as tf
def init_tensorflow():
    global tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    devices = tf.config.experimental.list_physical_devices()
    logger.info("Available devices:\n{devs}".format(
        devs=textwrap.indent(pprint.pformat(devices), prefix='  '),
    ))
    cpu_devices = tf.config.experimental.list_physical_devices('CPU')
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for gpu_device in gpu_devices:
        logger.info(f"ENABLE MEMORY GROWTH: {gpu_device}")
        tf.config.experimental.set_memory_growth(gpu_device, True)
# init_tensorflow()

# def import_tensorflow():
#     logger.info("Delayed import of tensorflow")
#     # Delay import to interact better with multiprocessing.
#     global tf
#     if tf is not None:
#         return
#     # Output logging information about operator placement on CPU / GPU.
#     # https://www.tensorflow.org/guide/gpu#logging_device_placement
#     # NOTE: Looks like these statements is ignored if ConfigProto is created manually
#     # (as it is in stable-baselines)
#     # tf.debugging.set_log_device_placement(True)

tf = None
stable_baselines = None
stable_baselines_common = None
stable_baselines_common_tf_util = None
common_vec_env = None
utils = None
stable_baselines_iml = None
iml = None
pybullet_envs = None
highway_env = None
gym = None

import_tensorflow_LOADED = False


def import_tensorflow():
    global import_tensorflow_LOADED
    if import_tensorflow_LOADED:
        return

    global tf
    global stable_baselines
    global stable_baselines_common
    global stable_baselines_common_tf_util
    global common_vec_env
    global utils
    global stable_baselines_iml
    global iml
    global pybullet_envs
    global highway_env
    global gym

    import tensorflow as tf
    init_tensorflow()

    import stable_baselines
    import stable_baselines.common as stable_baselines_common
    import stable_baselines.common.tf_util as stable_baselines_common_tf_util
    # from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv
    import stable_baselines.common.vec_env as common_vec_env
    import utils
    import stable_baselines.iml as stable_baselines_iml
    import iml_profiler.api as iml
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

    # Fix for breaking change in v2.6.0
    if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
        sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
        stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

    import_tensorflow_LOADED = True

cuda = None
trt = None
TRT_LOGGER = None
import_trt_LOADED = False
def import_trt():
    global import_trt_LOADED
    if import_trt_LOADED:
        return
    global cuda
    global trt
    import pycuda.driver as cuda
    # # This import causes pycuda to automatically manage CUDA context creation and cleanup.
    import pycuda.autoinit
    import tensorrt as trt

    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    import_trt_LOADED = True

def import_all():
    import_tensorflow()
    import_trt()

# from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

# from stable_baselines.iml import wrap_pybullet, unwrap_pybullet

# import iml_profiler.api as iml

# # Fix for breaking change in v2.6.0
# if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
#     sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
#     stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

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

def add_subparser_arguments(subparser_args, args):
    for attr, value in vars(subparser_args).items():
        if hasattr(args, attr):
            raise RuntimeError("Main argument parser already has --{opt}, but {mode} subparser has conflicting --{opt}".format(
                opt=attr,
                mode=args.mode,
            ))
            # assert not hasattr(args, attr)
        setattr(args, attr, value)

def _fake_worker(task_id):
    print(f"HELLO WORLD from task_id={task_id}")
    return task_id
    # raise RuntimeError(f"error from task_id={task_id}")

def _fake_worker_barrier(task_id, barrier):
    print(f"task_id={task_id} arrives")
    while True:
        time.sleep(2)
    barrier.wait()
    return task_id
    # raise RuntimeError(f"error from task_id={task_id}")

def main():
    setup_logging()

    # num_tasks = 3
    # barrier = multiprocessing.Barrier(num_tasks + 1)
    # # NOTE: we CANNOT pickle a multiprocessing.Barrier, since it results in a
    # # deadlock before the child thread even starts...
    # # I have NO IDEA why.
    # # with ProcessPoolExecutor() as pool:
    # with ThreadPoolExecutor() as pool:
    #     results = []
    #     for task_id in range(num_tasks):
    #         logger.info(f"Launch fake_worker task_id={task_id}")
    #         results.append(pool.submit(_fake_worker_barrier, task_id, barrier))
    #     logger.info("Wait for tasks in parent")
    #     barrier.wait()
    #     logger.info("All tasks done")
    #     for task_id, future in enumerate(results):
    #         result = future.result()
    #         assert result == task_id
    #     sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False,
                        # choices=list(utils.ALGOS.keys())
                        )
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
                            'convert_trt',
                            'microbench_inference',
                            'microbench_simulator',
                            'microbench_inference_multiprocess',
                        ],
                        default='default')
    parser.add_argument('--directory', help='output directory')
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
    args, argv = parser.parse_known_args()

    if args.mode not in {'microbench_inference_multiprocess'}:
        import_all()
        if args.mode not in {'convert_trt'}:
            iml.add_iml_arguments(parser)
            iml.register_wrap_module(stable_baselines_iml.wrap_pybullet, stable_baselines_iml.unwrap_pybullet)
            args, argv = parser.parse_known_args()

    if args.mode == 'microbench_inference':
        subparser = argparse.ArgumentParser("Microbenchmark TensorFlow inference throughput/latency on random data.")
        subparser.add_argument('--batch-size', type=int, default=1, help="Number of random samples per minibatch")
        subparser.add_argument('--warmup-iters', type=int, default=100)
        subparser_args = subparser.parse_args(argv)
        add_subparser_arguments(subparser_args, args)
    elif args.mode == 'microbench_inference_multiprocess':
        subparser = argparse.ArgumentParser("Microbenchmark TensorFlow inference throughput/latency on random data.")
        subparser.add_argument('--batch-size', type=int, default=1, help="Number of random samples per minibatch")
        subparser.add_argument('--warmup-iters', type=int, default=100)
        subparser.add_argument('--num-tasks', type=int, required=True, help="Number of parallel processes to perform inference")
        subparser.add_argument('--cpu', action='store_true', help="Use CPU for neural-network operators (default: GPU)")
        subparser.add_argument('--graph-def-pb', help="File containing serialized TensorFlow GraphDef protobuf; --algo and --env are ignored in this case (i.e., not used to lookup algorithm and environment)")
        subparser_args = subparser.parse_args(argv)
        add_subparser_arguments(subparser_args, args)
    elif args.mode == 'convert_trt':
        subparser = argparse.ArgumentParser("Convert TensorFlow GraphDef to TensorRT model.")
        subparser.add_argument('--trt-precision', required=True, choices=['fp32', 'fp16', 'int8'], help="Precision")
        subparser.add_argument('--trt-max-batch-size', required=True, type=int, help="Max batch size for profiling")
        subparser.add_argument('--graph-def-pb', help="File containing serialized TensorFlow GraphDef protobuf; --algo and --env are ignored in this case (i.e., not used to lookup algorithm and environment)")
        subparser.add_argument('--saved-model-dir', help="Directory containing serialized TensorFlow SavedModel")
        subparser_args = subparser.parse_args(argv)
        add_subparser_arguments(subparser_args, args)

    trained_agent = TrainedAgent(args)
    try:
        trained_agent.run()
    except BadArgumentException as e:
        parser.error(e.msg)
        sys.exit(1)

class BadArgumentException(RuntimeError):
    pass

class TrainedAgent:
    def __init__(self, args):
        self.args = args

    def error(self, msg):
        raise BadArgumentException(msg)

    def is_atari(self):
        args = self.args
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

    def make_env(self):
        args = self.args

        log_dir = self.log_dir()

        # Going through custom gym packages to let them register in the global registory
        for env_module in args.gym_packages:
            importlib.import_module(env_module)

        env_id = args.env
        algo = args.algo

        log_path = self.log_path()


        if algo in ['dqn', 'ddpg', 'sac']:
            args.n_envs = 1

        # from stable_baselines.iml import wrap_pybullet, unwrap_pybullet
        # from stable_baselines.common import set_global_seeds
        stable_baselines_common.set_global_seeds(args.seed)

        is_atari = self.is_atari()

        stats_path = os.path.join(log_path, env_id)
        hyperparams, stats_path = utils.get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

        env = utils.create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
                              stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                              should_render=not args.no_render,
                              hyperparams=hyperparams)
        return env

    def log_path(self):
        args = self.args

        env_id = args.env
        algo = args.algo
        folder = args.folder

        if args.exp_id == 0:
            args.exp_id = utils.get_latest_run_id(os.path.join(folder, algo), env_id)
            print('Loading latest experiment, id={}'.format(args.exp_id))

        # Sanity checks
        if args.exp_id > 0:
            log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
        else:
            log_path = os.path.join(folder, algo)

        return log_path


    def make_model(self, env):
        args = self.args

        log_path = self.log_path()

        algo = args.algo
        env_id = args.env

        model_path = "{}/{}.pkl".format(log_path, env_id)

        assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
        assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, env_id, model_path)

        # ACER raises errors because the environment passed must have
        # the same number of environments as the model was trained on.
        load_env = None if algo == 'acer' else env
        model = utils.ALGOS[algo].load(model_path, env=load_env)
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

    def mode_convert_trt(self):
        # from tensorflow.core.framework.graph_pb2 import GraphDef

        args = self.args

        def _output_trt_graph(trt,
                              sess, graph_def, output_names, trt_model_path,
                              trt_max_batch_size=8,
                              trt_precision='fp32'):
            assert trt_max_batch_size is not None
            # This is causing memory corruption...
            trt_out_graph = trt.create_inference_graph(
                input_graph_def=graph_def,
                outputs=output_names,
                max_batch_size=trt_max_batch_size,
                max_workspace_size_bytes=1 << 29,
                precision_mode=trt_precision)
            save_tensorflow_graph(sess, trt_out_graph, output_names, trt_model_path)
            logger.info("EXIT")
            sys.exit(0)
            # converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir)
            # converter.convert()
            # converter.save(output_saved_model_dir)

        def tfv1_output_trt_graph(sess, graph_def, output_names, trt_model_path,
                             trt_max_batch_size=8,
                             trt_precision='fp32'):
            import tensorflow.contrib.tensorrt as trt
            _output_trt_graph(trt,
                              sess, graph_def, output_names, trt_model_path,
                              trt_max_batch_size=trt_max_batch_size,
                              trt_precision=trt_precision,
                              )

        def tfv2_output_trt_graph(sess, graph_def, output_names, trt_model_path,
                trt_max_batch_size=8,
                trt_precision='fp32'):
            from tensorflow.python.compiler.tensorrt import trt_convert as trt
            _output_trt_graph(trt,
                              sess, graph_def, output_names, trt_model_path,
                              trt_max_batch_size=trt_max_batch_size,
                              trt_precision=trt_precision,
                              )

        def model_name(path):
            base = _b(path)
            name = base
            name = re.sub('\.pb$', '', name)
            name = re.sub('\.', '-', name)
            return name

        def trt_model_path(pb_path):
            base = "{name}.trt_precision_{prec}.trt_max_batch_size_{max_batch}.pb".format(
                name=model_name(pb_path),
                max_batch=args.trt_max_batch_size,
                prec=args.trt_precision,
            )
            return _j(_d(pb_path), base)

        if args.graph_def_pb is not None:
            graph = load_graph(args.graph_def_pb)
            graph_ctx = graph.as_default()
            graph_ctx.__enter__()
            self.sess = stable_baselines_common_tf_util.single_threaded_session(graph=graph)

            inputs, outputs = analyze_inputs_outputs(graph)
            output_names = [out.name for out in outputs]

            tfv2_output_trt_graph(
                self.sess, graph.as_graph_def(), output_names, trt_model_path(args.graph_def_pb),
                trt_max_batch_size=args.trt_max_batch_size,
                trt_precision=args.trt_precision)
        else:
            self.sess = stable_baselines_common_tf_util.single_threaded_session(graph=None)
            ret = load_saved_model(self.sess, args.saved_model_dir)
            import ipdb; ipdb.set_trace()

    def mode_save_tensorrt(self):
        args = self.args

        algo = args.algo

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

        algo = args.algo

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

    # @staticmethod
    # def _microbench_inference_multiprocess_worker(self):

    def mode_microbench_inference_multiprocess(self):
        args = self.args
        expr = MicrobenchInferenceMultiprocess(trained_agent=self, args=args)
        expr.run()

    def mode_microbench_inference(self):
        # import iml_profiler.api as iml
        args = self.args

        algo = args.algo

        # self.handle_iml(reports_progress=True)
        env = self.make_env()
        model = self.make_model(env)
        env_id = args.env

        is_atari = self.is_atari()

        obs = env.reset()

        # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
        deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her'] and not args.stochastic

        obs_space = env.observation_space
        # Q: Does observation_space already include the batch size...?

        # def random_minibatch(batch_size, obs_space):
        #     shape = (batch_size,) + obs_space.shape
        #     batch = np.random.uniform(size=shape)
        #     return batch

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
            self.error("You must use --iml-delay for --mode={mode}".format(mode=args.mode))

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
                if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, common_vec_env.VecEnv):
                    # DummyVecEnv
                    # Unwrap env
                    while isinstance(env, common_vec_env.VecNormalize) or isinstance(env, common_vec_env.VecFrameStack):
                        env = env.venv
                    env.envs[0].env.close()
                else:
                    # SubprocVecEnv
                    env.close()

    def mode_microbench_simulator(self):
        args = self.args

        if args.directory is None:
            self.error("output --directory required for --mode={mode}".format(
                mode=args.mode,
            ))

        algo = args.algo
        # Q: How to ENSURE that we are running a SINGLE simulator instance (i.e., no subprocess crap)

        if args.n_envs != 1:
            self.error("For --mode={mode} you must use --n-envs=1 but saw {n_envs}".format(
                n_envs=args.n_envs,
                mode=args.mode,
            ))
        env = self.make_env()
        assert not isinstance(env, common_vec_env.SubprocVecEnv)
        if isinstance(env, common_vec_env.DummyVecEnv) or isinstance(env, common_vec_env.VecFrameStack):
            assert len(env.envs) == 1
        assert isinstance(env, common_vec_env.VecEnv) and env.num_envs == 1
        # model = self.make_model()
        env_id = args.env

        is_atari = self.is_atari()

        obs = env.reset()

        # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
        deterministic = self.deterministic()

        step_time_sec = []
        iterations_start_t = time.time()
        for i in range(args.iterations):
            # action, _ = model.predict(obs, deterministic=deterministic)

            # Random Agent
            action = [env.action_space.sample()]
            # Clip Action to avoid out of bound errors
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            start_t = time.time()
            obs, reward, done, infos = env.step(action)
            end_t = time.time()
            time_sec = end_t - start_t
            step_time_sec.append(time_sec)

            # episode_reward += reward[0]
            # ep_len += 1
        iterations_end_t = time.time()

        # iterations_total_sec = iterations_end_t - iterations_start_t
        # time_sec_per_iteration = iterations_total_sec / args.iterations

        # Metrics we want:
        # - Throughput (samples per second).
        #   - Mean
        #   - Stdev
        # - Latency (seconds per sample)
        #   - Mean
        #   - Stdev

        json_path = _j(args.directory, 'mode_microbench_simulator.json')
        js = JsonFile(json_path, mode='w')
        step_time_sec = np.array(step_time_sec)

        js['raw_samples'] = dict()
        js['raw_samples']['step_time_sec'] = step_time_sec.tolist()

        js['metadata'] = dict()
        js['metadata']['iterations'] = args.iterations
        js['metadata']['env'] = args.env

        total_steps = len(step_time_sec)
        total_time_sec = np.sum(step_time_sec)
        js['summary_metrics'] = dict()
        js['summary_metrics']['env'] = args.env
        js['summary_metrics']['total_steps'] = total_steps
        js['summary_metrics']['total_time_sec'] = total_time_sec
        js['summary_metrics']['mean_step_time_sec'] = step_time_sec.mean()
        js['summary_metrics']['stdev_step_time_sec'] = step_time_sec.std()
        js['summary_metrics']['throughput_step_per_sec'] = total_steps / total_time_sec

        # js.append('iterations_total_sec', iterations_total_sec)
        # js.append('iterations', args.iterations)
        # js.append('time_sec_per_iteration', time_sec_per_iteration)
        # js['repetitions'] = js.get('repetitions', 0) + 1

        print("> Dump --mode={mode} results @ {path}".format(
            path=json_path,
            mode=args.mode,
        ))
        js.dump()

        # Workaround for https://github.com/openai/gym/issues/893
        if not args.no_render:
            if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, common_vec_env.VecEnv):
                # DummyVecEnv
                # Unwrap env
                while isinstance(env, common_vec_env.VecNormalize) or isinstance(env, common_vec_env.VecFrameStack):
                    env = env.venv
                env.envs[0].env.close()
            else:
                # SubprocVecEnv
                env.close()


    def mode_default(self):
        args = self.args

        algo = args.algo

        # self.handle_iml(reports_progress=False)
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
                if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, common_vec_env.VecEnv):
                    # DummyVecEnv
                    # Unwrap env
                    while isinstance(env, common_vec_env.VecNormalize) or isinstance(env, common_vec_env.VecFrameStack):
                        env = env.venv
                    env.envs[0].env.close()
                else:
                    # SubprocVecEnv
                    env.close()

    def deterministic(self):
        args = self.args
        algo = args.algo
        # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
        deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her'] and not args.stochastic
        return deterministic

    def mode_microbench_iml_python_annotation(self):
        args = self.args

        # self.handle_iml(reports_progress=True)

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

        algo = args.algo

        # self.handle_iml(reports_progress=True)
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
                if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, common_vec_env.VecEnv):
                    # DummyVecEnv
                    # Unwrap env
                    while isinstance(env, common_vec_env.VecNormalize) or isinstance(env, common_vec_env.VecFrameStack):
                        env = env.venv
                    env.envs[0].env.close()
                else:
                    # SubprocVecEnv
                    env.close()

    def mode_microbench_iml_clib_interception_tensorflow(self):
        args = self.args

        algo = args.algo

        # self.handle_iml(reports_progress=True)
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
                if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, common_vec_env.VecEnv):
                    # DummyVecEnv
                    # Unwrap env
                    while isinstance(env, common_vec_env.VecNormalize) or isinstance(env, common_vec_env.VecFrameStack):
                        env = env.venv
                    env.envs[0].env.close()
                else:
                    # SubprocVecEnv
                    env.close()

    def run(self):
        args = self.args

        if args.mode == 'default':
            handle_iml(trained_agent=self, reports_progress=False)
            self.mode_default()
        elif args.mode == 'microbench_iml_python_annotation':
            handle_iml(trained_agent=self, reports_progress=True)
            self.mode_microbench_iml_python_annotation()
        elif args.mode == 'microbench_iml_clib_interception_simulator':
            handle_iml(trained_agent=self, reports_progress=True)
            self.mode_microbench_iml_clib_interception_simulator()
        elif args.mode == 'microbench_iml_clib_interception_tensorflow':
            handle_iml(trained_agent=self, reports_progress=True)
            self.mode_microbench_iml_clib_interception_tensorflow()
        elif args.mode == 'save_tensorrt':
            self.mode_save_tensorrt()
        elif args.mode == 'convert_trt':
            self.mode_convert_trt()
        elif args.mode == 'load_tensorrt':
            self.mode_load_tensorrt()
        elif args.mode == 'microbench_inference':
            handle_iml(trained_agent=self, reports_progress=True)
            self.mode_microbench_inference()
        elif args.mode == 'microbench_simulator':
           self.mode_microbench_simulator()
        elif args.mode == 'microbench_inference_multiprocess':
            self.mode_microbench_inference_multiprocess()
        else:
            raise NotImplementedError("Note sure how to run --mode={mode}".format(
                mode=args.mode))


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

def random_minibatch(batch_size, obs_space=None, shape=None):
    if shape is None:
        shape = obs_space.shape
    shape = (batch_size,) + tuple(shape)
    batch = np.random.uniform(size=shape)
    return batch


class BaseInferenceWorker:
    def __init__(self, expr, task_id):
        self.expr = expr
        self.task_id = task_id

    def infer_minibatch(self):
        raise NotImplementedError()

    def deinit(self):
        pass

    def run(self):
        args = self.expr.args
        task_id = self.task_id
        logger.info(f"Start worker task_id={task_id}")
        do_with_device(args, self._run)

    def _run(self):
        args = self.expr.args
        expr = self.expr
        task_id = self.task_id
        def is_warmed_up(t):
            return t > args.warmup_iters

        warmed_up = False
        inference_time_sec = []
        for t in range(args.n_timesteps):

            if not warmed_up and is_warmed_up(t):
                # Entire training loop is now running; enable IML tracing
                start_t = time.time()
                start_timesteps = t
                if expr.barrier is not None:
                    logger.info(expr.log_msg(f"Await inference in thread={task_id}..."))
                    expr.barrier.wait()
                    logger.info(expr.log_msg(f"Start inference in thread={task_id}"))
                else:
                    pass
                    logger.info(expr.log_msg(f"Running inference in single-process mode in thread={task_id}"))
                warmed_up = True

            start_inference_t = time.time()
            output = self.infer_minibatch()
            # action, _ = model.predict(random_obs, deterministic=deterministic)
            end_inference_t = time.time()
            if warmed_up:
                inf_time_sec = end_inference_t - start_inference_t
                inference_time_sec.append(inf_time_sec)

        end_t = time.time()

        if expr.num_tasks == 1:
            expr.single_process_start_t = start_t
            expr.single_process_end_t = end_t

        inference_time_sec = np.array(inference_time_sec)
        # total_time_sec = (end_t - start_t)
        total_time_sec = np.sum(inference_time_sec)
        # total_samples = (args.n_timesteps - start_timesteps)*args.batch_size
        total_samples = len(inference_time_sec)*args.batch_size
        throughput_qps = total_samples / total_time_sec
        inference_js = dict()
        # Q: Should difference InferenceWorker's add extra fields...?
        inference_js['raw_samples'] = dict()
        inference_js['summary_metrics'] = dict()
        inference_js['raw_samples']['inference_time_sec'] = inference_time_sec.tolist()
        inference_js['summary_metrics']['throughput_qps'] = throughput_qps
        inference_js['summary_metrics']['total_samples'] = total_samples
        inference_js['summary_metrics']['batch_size'] = args.batch_size
        inference_js['summary_metrics']['total_time_sec'] = total_time_sec
        inference_js['summary_metrics']['mean_inference_time_sec'] = inference_time_sec.mean()
        inference_js['summary_metrics']['std_inference_time_sec'] = inference_time_sec.std()
        inference_js['summary_metrics']['inference_time_percentile_99_sec'] = np.percentile(inference_time_sec, 0.99)
        js_path = _j(args.directory, f"mode_microbench_inference_multiprocess.task_id_{task_id}.json")
        do_dump_json(inference_js, js_path)
        self.deinit()
        return js_path


class StableBaselinesInferenceWorker(BaseInferenceWorker):
    def __init__(self, expr, task_id):
        super().__init__(expr, task_id)

        expr = self.expr
        args = expr.args
        algo = args.algo
        trained_agent = expr.trained_agent

        self.env = trained_agent.make_env()
        self.model = trained_agent.make_model(self.env)
        self.env_id = args.env

        self.is_atari = trained_agent.is_atari()

        obs = self.env.reset()

        # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
        self.deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her'] and not args.stochastic

        obs_space = self.env.observation_space

        process_name = get_process_name(args)
        phase_name = process_name
        self.random_obs = random_minibatch(args.batch_size, obs_space)
        logger.info(expr.log_msg("using --batch-size={batch} => {shape}".format(
            batch=args.batch_size,
            shape=self.random_obs.shape,
        )))

    def infer_minibatch(self):
        action, _ = self.model.predict(self.random_obs, deterministic=self.deterministic)
        return action

    def deinit(self):
        args = self.expr.args
        # Workaround for https://github.com/openai/gym/issues/893
        if not args.no_render:
            if args.n_envs == 1 and 'Bullet' not in self.env_id and not self.is_atari and isinstance(self.env, common_vec_env.VecEnv):
                # DummyVecEnv
                # Unwrap env
                while isinstance(self.env, common_vec_env.VecNormalize) or isinstance(self.env, common_vec_env.VecFrameStack):
                    self.env = self.env.venv
                self.env.envs[0].env.close()
            else:
                # SubprocVecEnv
                self.env.close()


class GraphDefInferenceWorker(BaseInferenceWorker):
    def __init__(self, expr, task_id):
        super().__init__(expr, task_id)

        expr = self.expr
        args = expr.args

        # TODO: read tensorflow graph from --graph-def-pb, lookup input nodes and shape,
        # create random numpy observation of matching shape (with first -1 dimension
        # replaced with args.batch_size).

        assert args.graph_def_pb is not None
        self.graph = load_graph(args.graph_def_pb)
        self.inputs, self.outputs = analyze_inputs_outputs(self.graph)

        # assert len(self.inputs) == 1
        self.random_inputs = []
        self.input_placeholders = []
        self.feed_dict = dict()
        for i, inp in enumerate(self.inputs):
            shape = operation_shape(inp)
            logger.info(f"Generate random input {i}: batch_size={args.batch_size} shape={shape}")
            # tf.placeholder(np.float32, shape = [None, 32, 32, 3], name='input')
            random_input = random_minibatch(args.batch_size, shape=shape)
            assert len(inp.values()) == 1
            placeholder = inp.values()[0]
            self.random_inputs.append(random_input)
            self.feed_dict[placeholder] = random_input

        # with self.graph.as_default():
        #     self.sess = tf_util.single_threaded_session(graph=self.graph)

        self.graph_ctx = self.graph.as_default()
        self.graph_ctx.__enter__()
        self.sess = stable_baselines_common_tf_util.single_threaded_session(graph=self.graph)

        # self.sess = tf.compat.v1.get_default_session()
        assert self.sess is not None

        # import ipdb; ipdb.set_trace()

        # with self.graph.as_default():
        #     self.sess = tf_util.single_threaded_session(graph=self.graph)


        # logger.info(expr.log_msg("using --batch-size={batch} => {shape}".format(
        #     batch=args.batch_size,
        #     shape=self.random_obs.shape,
        # )))


    def infer_minibatch(self):
        outputs = self.sess.run(self.outputs, feed_dict=self.feed_dict)
        return outputs

    def deinit(self):
        self.graph_ctx.__exit__()


class MicrobenchInferenceMultiprocess:
    def __init__(self, trained_agent, args):
        self.trained_agent = trained_agent
        self.args = args
        # self.failed = None
        self.failed = multiprocessing.Event()

    def run(self):
        args = self.args

        if args.directory is None:
            self.error("output --directory required for --mode={mode}".format(
                mode=args.mode,
            ))

        if args.warmup_iters > args.n_timesteps is None:
            self.error("--warmup-iters={warmup_iters} must be less than --n-timesteps={n_timesteps}".format(
                warmup_iters=args.warmup_iters,
                n_timesteps=args.n_timesteps,
            ))

        def with_Process():
            if 'tensorflow' in sys.modules.keys():
                logger.info("Python modules loaded BEFORE forking children:\n{modules}".format(
                    modules=textwrap.indent(pprint.pformat(sorted(sys.modules.keys())), prefix='  '),
                ))
                assert 'tensorflow' not in sys.modules.keys()
            if self.num_tasks > 1:
                # +1 for parent thread.
                self.barrier = multiprocessing.Barrier(self.num_tasks + 1)

                procs = []
                for task_id in range(self.num_tasks):
                    logger.info(self.log_msg(f"Launch child task_id={task_id}"))
                    proc = multiprocessing.Process(target=MicrobenchInferenceMultiprocess._worker, args=(self, task_id))
                    proc.start()
                    procs.append(proc)

                # Wait for warmup period in each worker.
                logger.info(self.log_msg(f"Await start of inference in parent..."))
                failed = False
                try:
                    self.barrier.wait()
                    # self.barrier.wait(timeout=2)
                except threading.BrokenBarrierError as e:
                    # self.error(f"Parent saw broken barrier from at least one failed child", exitcode=1, exception=e)
                    logging.error(f"Parent saw broken barrier from at least one failed child; waiting for children")
                    for proc in procs:
                        proc.join()
                    failed = True
                if failed:
                    logging.error(f"Exit parent with exitcode=1")
                    sys.exit(1)
                logger.info(self.log_msg(f"Start inference in parent..."))

                start_t = time.time()

                # Wait for all the processes to finish writing their json files.
                # TODO: poll on processes; if at least one fails with bad exit code, then check "failed" condition.
                # (Wait for all processes to finish)

                # finished_procs = []
                # alive_procs = list(enumerate(procs))
                # while len(alive_procs) > 0:
                #     i = 0
                #     while i < len(alive_procs):
                #         task_id, proc = alive_procs[i]
                #         if not proc.is_alive():
                #             # finished_proc = alive_procs[i]
                #             del alive_procs[i]
                #             assert proc.exitcode is not None
                #             if proc.exitcode < 0:
                #                 self.error(f"Parent saw failed task_id={task_id}")
                #             finished_procs.append((task_id, proc))
                #         else:
                #             i += 1
                #     time.sleep(2)

                logger.info(self.log_msg(f"Await end of inference in parent..."))
                for task_id, proc in enumerate(procs):
                    proc.join()
                    assert proc.exitcode is not None
                    if proc.exitcode < 0:
                        self.error(f"Parent saw failed task_id={task_id}")
                    # js_path = future.result()
                    # js_paths.append(js_path)

                end_t = time.time()
                logger.info(self.log_msg(f"Saw end of inference in parent (took {end_t - start_t} sec)"))

            else:
                self.barrier = None
                assert self.num_tasks == 1
                self.worker(task_id=0)
                start_t = self.single_process_start_t
                end_t = self.single_process_end_t
                # end_t = time.time()

            js_paths = [path for path in each_file_recursive(args.directory) \
                        if re.search(r'^mode_microbench_inference_multiprocess.task_id_\d+.json$', _b(path))]

            return start_t, end_t, js_paths

        # js_paths = with_pool()
        start_t, end_t, js_paths = with_Process()
        logger.info(self.log_msg("Parent saw result from children:\n{paths}".format(
            paths=textwrap.indent(pprint.pformat(js_paths), prefix='  '),
        )))
        assert len(js_paths) > 0

        # Compute:
        # - throughput (overall across all processes)
        #   - inference_time_sec = concat inference_js['raw_samples']['inference_time_sec'] from all processes
        #   - total_time_sec     = end_t - start_t
        # - latency samples (mean/std over all processes)
        #   - inference_time_sec = concat inference_js['raw_samples']['inference_time_sec'] from all processes
        #   - compute mean/std
        total_time_sec = end_t - start_t
        js_datas = [load_json(path) for path in js_paths]
        inference_time_sec = np.array(list(itertools.chain.from_iterable(js['raw_samples']['inference_time_sec'] for js in js_datas)))
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
        inference_js['summary_metrics']['std_inference_time_sec'] = inference_time_sec.std()
        inference_js['summary_metrics']['inference_time_percentile_99_sec'] = np.percentile(inference_time_sec, 0.99)
        js_path = _j(args.directory, 'mode_microbench_inference_multiprocess.merged.json')
        logger.info(self.log_msg("Dump {path}".format(path=js_path)))
        do_dump_json(inference_js, js_path)

    # def _worker(self, *args, **kwargs):
    @staticmethod
    def _worker(self, task_id):
        try:
            # print(f"HELLO WORLD from task_id={task_id}")
            # logger.info(f"Start worker task_id={task_id}")
            # return self.worker(*args, **kwargs)
            return self.worker(task_id)
        except Exception as e:
            self.failed.set()
            self.barrier.abort()
            tb = traceback.format_exc()
            logger.info("Saw exception in task_id={task_id}:\n{msg}".format(
                task_id=task_id,
                msg=textwrap.indent(str(tb), prefix='  ')
            ))
            raise e

    # def _fake_worker(task_id):
    @staticmethod
    def _fake_worker(self, task_id):
        print(f"HELLO WORLD from task_id={task_id}")
        return task_id
        # raise RuntimeError(f"error from task_id={task_id}")

    def worker(self, task_id):
        # import_trt()
        import_tensorflow()
        assert tf is not None
        # import_all()

        args = self.args

        if args.graph_def_pb is not None:
            InferenceWorkerKlass = GraphDefInferenceWorker
        else:
            InferenceWorkerKlass = StableBaselinesInferenceWorker
        inference_worker = InferenceWorkerKlass(expr=self, task_id=task_id)
        inference_worker.run()

    @property
    def num_tasks(self):
        return self.args.num_tasks

    def error(self, msg, exitcode=None, exception=None):
        if exitcode is not None:
            if exception is not None:
                tb = traceback.format_exc()
                logger.error("Saw exception:\n{msg}".format(
                    msg=textwrap.indent(str(tb), prefix='  ')
                ))
            logger.error(msg)
            logger.error(f"Exit parent with exitcode={exitcode}")
            sys.exit(exitcode)
        if exception is None:
            exception = RuntimeError(msg)
        raise exception

    def log_msg(self, msg):
        return log_msg('INFER', msg)

def each_file_recursive(root_dir):
    if not os.path.isdir(root_dir):
        raise ValueError("No such directory {root_dir}".format(root_dir=root_dir))
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for base in filenames:
            path = _j(dirpath, base)
            yield path

def handle_iml(trained_agent, reports_progress):
    import iml_profiler.api as iml
    args = trained_agent.args
    parser = trained_agent.parser
    iml_directory = trained_agent.get_iml_directory()
    iml.handle_iml_args(parser, args, directory=iml_directory, reports_progress=reports_progress)
    iml.prof.set_metadata({
        'algo': args.algo,
        'env': args.env,
    })

def do_with_device(args, func):
    devices = tf.config.experimental.list_physical_devices()
    logger.info("Available devices:\n{devs}".format(
        devs=textwrap.indent(pprint.pformat(devices), prefix='  '),
    ))
    cpu_devices = tf.config.experimental.list_physical_devices('CPU')
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')

    if args.cpu:
        if len(cpu_devices) < 1:
            raise RuntimeError("Couldn't allocate CPU since none are available; devices available are:\n{devs}".format(
                devs=textwrap.indent(pprint.pformat(devices), prefix='  '),
            ))
        device = cpu_devices[0]
    else:
        if len(gpu_devices) < 1:
            raise RuntimeError("Couldn't allocate GPU since none are available; devices available are:\n{devs}".format(
                devs=textwrap.indent(pprint.pformat(devices), prefix='  '),
            ))
        device = gpu_devices[0]

    logger.info(f"Running with device={device}")
    device_name = re.sub(r'physical_device', 'device', device.name)
    # device_name = "/device:GPU:0"
    with tf.device(device_name):
        func()


def load_saved_model(sess, saved_model_dir):
    # return tf.saved_model.load(sess, ["master"], saved_model_dir)
    return tf.saved_model.load(saved_model_dir, ["master"])
    # with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
    #     graph_def = tf.compat.v1.GraphDef()
    #     graph_def.ParseFromString(f.read())
    # with tf.Graph().as_default() as graph:
    #     tf.import_graph_def(graph_def)
    # return graph


def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def save_tensorflow_graph(sess, graph, output_names, model_path):
    from tensorflow.core.framework.graph_pb2 import GraphDef
    if type(graph) == GraphDef:
        graph_def = graph
    else:
        assert type(graph) == tf.Graph
        graph_def = graph.as_graph_def()
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess, graph_def,
        output_names,
    )
    frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)

    logger.info("Write TensorFlow inference graph to {path}".format(
        path=model_path))
    with open(model_path, "wb") as f:
        f.write(frozen_graph.SerializeToString())

def operation_shape(tf_op, skip_batch_dim=True):
    shape = [dim.size for dim in tf_op.node_def.attr['shape'].shape.dim]
    if shape[0] == -1 and skip_batch_dim:
        shape = shape[1:]
    return shape

def analyze_inputs_outputs(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if len(op.inputs) == 0 and op.type != 'Const':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)
    return (inputs, outputs)

if __name__ == '__main__':
    main()
