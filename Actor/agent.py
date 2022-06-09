from hok.gamecore.kinghonour.agent import Agent as BaseAgent
import numpy as np

class Agent(BaseAgent):
    def __init__(self, model_cls, model_pool_addr, keep_latest=False, local_mode=False, rule_only=False):
        super().__init__(model_cls, model_pool_addr, keep_latest, local_mode, rule_only)

    def _predict_process(self, hero_data_list, frame_state, runtime_ids):
        pred_ret, lstm_info = super()._predict_process(
            hero_data_list, frame_state, runtime_ids
        )
        pred = pred_ret
        if np.random.random() > 0.95:
            pred = [np.random.random(pred_ret[i].shape) for i in range(3)]
        return pred, lstm_info