import sys
sys.path.append('.')
#sys.path.append('./lib')

from absl import app as absl_app
from absl import flags
import random
import threading

import sail.common.logging as LOG

#from actor import Actor
#from agent import RandomAgent as Agent
#from sample.sample_manager import DummySampleManager as SampleManager
# from predict.model import Model
from algorithms.model.model import Model
from algorithms.model.model_config import ModelConfig
from hok.actor import Actor
#from hok.gamecore.kinghonour.agent import Agent as Agent
from algorithms.agent.agent import Agent
from hok.algorithms.model.sample_manager import SampleManager as SampleManager

FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "actor id")
flags.DEFINE_integer("max_step", 500, "max step of one round")
flags.DEFINE_string("mem_pool_addr", "127.0.0.1:35200", "address of memory pool")
flags.DEFINE_string("model_pool_addr", "localhost:10016", "address of model pool")
flags.DEFINE_string("gc_server_addr", "127.0.0.1:23432", "address of gamecore server")
flags.DEFINE_string("ai_server_ip", "172.18.128.2", "host of ai_server")
# flags.DEFINE_string("infer_server_addr", "localhost:18000:8000", "address of remote inference server")
# flags.DEFINE_string("predictor_type", "local", "type of predictor, local or remote")
flags.DEFINE_integer("thread_num", 1, "thread_num")

flags.DEFINE_string("agent_models", "", "agent_model_list")
flags.DEFINE_integer("eval_number", -1, "battle number for evaluation")

AGENT_NUM = 2

# gamecore as lib
def gc_as_lib(argv):
    # TODO: used for different process
    from hok.gamecore.kinghonour.gamecore_client import GameCoreClient as Environment
    thread_id = 0
    actor_id = FLAGS.thread_num * FLAGS.actor_id + thread_id
    agents = []
    game_id_init = "None"
    main_agent = random.randint(0, 1)

    eval_number = FLAGS.eval_number
    load_models = FLAGS.agent_models.split(',')
    print(load_models)
    for i, m in enumerate(load_models):
        if m == "common_ai":
            load_models[i] = None
    eval_mode = eval_number > 0

    for i in range(AGENT_NUM):
        agents.append(Agent(
            Model(ModelConfig), FLAGS.model_pool_addr.split(";"),
            keep_latest=(i==main_agent and not eval_mode), local_mode=eval_mode))
    env = Environment(host=FLAGS.ai_server_ip, seed=actor_id, gc_server=FLAGS.gc_server_addr)
    sample_manager = SampleManager(mem_pool_addr=FLAGS.mem_pool_addr, mem_pool_type="mcp++",
                                   num_agents=AGENT_NUM, game_id=game_id_init, local_mode=eval_mode)
    actor = Actor(id=actor_id, agents=agents, )
    actor.set_sample_managers(sample_manager)
    actor.set_env(env)
    actor.run(eval_mode=eval_mode, eval_number=eval_number, load_models=load_models)

if __name__ == '__main__':
    absl_app.run(gc_as_lib)
