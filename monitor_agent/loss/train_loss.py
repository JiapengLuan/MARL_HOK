# shall be put in /data1/monitor_agent/process

import sys
sys.dont_write_bytecode = True
import subprocess
sys.path.append("..")
from common import PLogger
from process import monitor_process

logger = PLogger.Logging.GetLogger()
class train_loss(monitor_process.monitor_process):
    def __init__(self, monitor_param, monitor_key):
        self.key_list = monitor_key.split(',')
        self.param = monitor_param

def run(self):
    ret, msg = self.process()

    if ret < 0:
        logger.error("process process error, ret:%s, msg:%s" %(ret, msg))
    else:
        logger.info("process ret [%s] msg [%s]" %(ret, msg))
    msg = msg.split(',')

    json_list = []
    json_field = {}
    json_tag = {}
    json_temp = {}
    for idx, val in enumerate(msg):
        key = self.key_list[idx]
        json_field[key] = float(val)
    json_temp['fields'] = json_field
    json_temp['tags'] = json_tag
    json_list.append(json_temp)

    return ret, json_list

def process(self):
    ret, msg = subprocess.getstatusoutput("bash ../script/train_loss.sh")
    return ret, msg
