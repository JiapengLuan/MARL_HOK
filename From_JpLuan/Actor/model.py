#__author__ = "yannickyu"

import tensorflow as tf
import glob
import numpy as np
import h5py
from algorithms.model.model_config import ModelConfig as Config
import sys
import threading


class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args):
        if self._cls not in self._instance:
            with Singleton._instance_lock:
                if self._cls not in self._instance:
                    self._instance[self._cls] = self._cls(*args)
        return self._instance[self._cls]


# 添加装饰器，将Model的实例化变成单例模式
@Singleton
class Model():

    def __init__(self, Config, use_xla=False):
        # feature configure parameter
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.hero_data_split_shape = Config.HERO_DATA_SPLIT_SHAPE
        self.hero_seri_vec_split_shape = Config.HERO_SERI_VEC_SPLIT_SHAPE
        self.hero_label_size_list = Config.HERO_LABEL_SIZE_LIST
        self.hero_is_reinforce_task_list = Config.HERO_IS_REINFORCE_TASK_LIST

        self.learning_rate = Config.INIT_LEARNING_RATE_START
        self.var_beta = Config.BETA_START

        self.clip_param = Config.CLIP_PARAM
        self.each_hero_loss_list = []
        self.min_policy = Config.MIN_POLICY

        self.hero_num = Config.HERO_NUM
        self.hero_data_len = sum(Config.data_shapes[0])

        # for actor
        self.lstm_time_steps = 1
        self.batch_size = 1

        self.feature_img_dim = Config.HERO_SERI_VEC_SPLIT_SHAPE[0][0]
        self.feature_vec_dim = Config.HERO_SERI_VEC_SPLIT_SHAPE[0][1]
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.graph = None

    def get_input_tensors(self):
        # return [self.feature_ph, self.legal_action_ph, self.lstm_cell_ph, self.lstm_hidden_ph]
        return [self.feature_hero0, self.feature_hero1, self.feature_hero2, self.lstm_cell_ph, self.lstm_hidden_ph]

    def get_output_tensors(self):
        return [self.probs_h0, self.probs_h1, self.probs_h2, self.lstm_cell_output, self.lstm_hidden_output]

    def build_infer_graph(self):
        if self.graph is not None:
            return self.graph
        self.graph = self._build_infer_graph()
        return self.graph

    def _build_infer_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            hero_data_list = []
            hero_lstm_cell_list = []
            hero_lstm_hidden_list = []

            self.feature_hero0 = tf.placeholder(shape=(
                self.batch_size, self.hero_data_split_shape[0][0]), name="feature_hero0", dtype=np.float32)
            hero_data_list.append(self.feature_hero0)
            self.feature_hero1 = tf.placeholder(shape=(
                self.batch_size, self.hero_data_split_shape[1][0]), name="feature_hero1", dtype=np.float32)
            hero_data_list.append(self.feature_hero1)
            self.feature_hero2 = tf.placeholder(shape=(
                self.batch_size, self.hero_data_split_shape[2][0]), name="feature_hero2", dtype=np.float32)
            hero_data_list.append(self.feature_hero2)
            self.lstm_cell_ph = tf.placeholder(shape=(
                self.batch_size, self.lstm_hidden_dim), name="lstm_cell", dtype=np.float32)
            self.lstm_hidden_ph = tf.placeholder(shape=(
                self.batch_size, self.lstm_hidden_dim), name="lstm_hidden", dtype=np.float32)

            each_hero_data_list = self._split_inference_data(hero_data_list)
            probs_list = self._inference(
                each_hero_data_list, only_inference=True)
            # 3v3 output
            self.probs_h0 = tf.layers.flatten(tf.concat(probs_list[0], axis=1))
            self.probs_h1 = tf.layers.flatten(tf.concat(probs_list[1], axis=1))
            self.probs_h2 = tf.layers.flatten(tf.concat(probs_list[2], axis=1))
            self.lstm_cell_output = self.lstm_cell_ph
            self.lstm_hidden_output = self.lstm_hidden_ph

            self.init = tf.global_variables_initializer()
        return self.graph

    def build_graph(self, datas, global_step=None):
        datas = tf.reshape(datas, [-1, self.hero_num, self.hero_data_len])
        data_list = tf.transpose(datas, perm=[1, 0, 2])

        # new add 20180912
        each_hero_data_list = []
        for hero_index in range(self.hero_num):
            this_hero_data_list = self._split_data(tf.cast(
                data_list[hero_index], tf.float32), self.hero_data_split_shape[hero_index])
            each_hero_data_list.append(this_hero_data_list)
        # build network
        each_hero_fc_result_list = self._inference(each_hero_data_list)
        # calculate loss
        with tf.xla.experimental.jit_scope(self.use_xla):
            loss = self._calculate_loss(
                each_hero_data_list, each_hero_fc_result_list)
        # return
        return loss, [loss, [self.each_hero_loss_list[0][1], self.each_hero_loss_list[0][2], self.each_hero_loss_list[0][3]], [self.each_hero_loss_list[1][1], self.each_hero_loss_list[1][2], self.each_hero_loss_list[1][3]], [self.each_hero_loss_list[2][1], self.each_hero_loss_list[2][2], self.each_hero_loss_list[2][3]]]

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.00001)

    def _split_inference_data(self, hero_data_list):
        # new add 20180912
        each_hero_data_list = []
        for hero_index in range(self.hero_num):
            this_hero_data = []
            data_feature = tf.cast(hero_data_list[hero_index], tf.float32)
            print(data_feature.shape)
            print(self.hero_data_split_shape[hero_index][0])
            feature = tf.reshape(
                data_feature, [-1, self.hero_data_split_shape[hero_index][0]])
            this_hero_data.append(feature)
            each_hero_data_list.append(this_hero_data)

        return each_hero_data_list

    def _split_data(self, this_hero_data, this_hero_data_split_shape):
        # calculate length of each frame
        this_hero_each_frame_data_length = np.sum(
            np.array(this_hero_data_split_shape))
        this_hero_sequence_data_length = this_hero_each_frame_data_length * self.lstm_time_steps
        this_hero_sequence_data_split_shape = np.array(
            [this_hero_sequence_data_length, self.lstm_unit_size, self.lstm_unit_size])
        sequence_data, lstm_cell_data, lstm_hidden_data = tf.split(
            this_hero_data, this_hero_sequence_data_split_shape, axis=1)
        reshape_sequence_data = tf.reshape(
            sequence_data, [-1, this_hero_each_frame_data_length])
        sequence_this_hero_data_list = tf.split(
            reshape_sequence_data, np.array(this_hero_data_split_shape), axis=1)
        self.lstm_cell_ph = lstm_cell_data
        self.lstm_hidden_ph = lstm_hidden_data
        return sequence_this_hero_data_list

    def _calculate_loss(self, each_hero_data_list, each_hero_fc_result_list):
        self.cost_all = tf.constant(0.0, dtype=tf.float32)
        for hero_index in range(len(each_hero_data_list)):
            this_hero_label_task_count = len(
                self.hero_label_size_list[hero_index])
            this_hero_legal_action_flag_list = each_hero_data_list[hero_index][1:(
                1+this_hero_label_task_count)]
            this_hero_reward_list = each_hero_data_list[hero_index][(
                1+this_hero_label_task_count):(2+this_hero_label_task_count)]
            this_hero_advantage = each_hero_data_list[hero_index][(
                2+this_hero_label_task_count)]
            this_hero_action_list = each_hero_data_list[hero_index][(
                3+this_hero_label_task_count):(3+this_hero_label_task_count*2)]
            this_hero_probability_list = each_hero_data_list[hero_index][(
                3+this_hero_label_task_count*2):(3+this_hero_label_task_count*3)]
            this_hero_frame_is_train = each_hero_data_list[hero_index][(
                3+this_hero_label_task_count*3)]
            this_hero_weight_list = each_hero_data_list[hero_index][(
                4+this_hero_label_task_count*3):(4+this_hero_label_task_count*4)]
            this_hero_fc_label_list = each_hero_fc_result_list[hero_index][:-1]
            this_hero_value = each_hero_fc_result_list[hero_index][-1]
            this_hero_all_loss_list = self._calculate_single_hero_loss(this_hero_legal_action_flag_list, this_hero_reward_list, this_hero_advantage, this_hero_action_list, this_hero_probability_list,
                                                                       this_hero_frame_is_train, this_hero_fc_label_list, this_hero_value, self.hero_label_size_list[hero_index], self.hero_is_reinforce_task_list[hero_index], this_hero_weight_list)
            self.cost_all = self.cost_all + this_hero_all_loss_list[0]
            self.each_hero_loss_list.append(this_hero_all_loss_list)
        return self.cost_all

    def _squeeze_tensor(self, unsqueeze_reward_list, unsqueeze_advantage, unsqueeze_label_list, unsqueeze_frame_is_train, unsqueeze_weight_list):
        reward_list = []
        for ele in unsqueeze_reward_list:
            reward_list.append(tf.squeeze(ele, axis=[1]))
        advantage = tf.squeeze(unsqueeze_advantage, axis=[1])
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(tf.squeeze(ele, axis=[1]))
        frame_is_train = tf.squeeze(unsqueeze_frame_is_train, axis=[1])
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(tf.squeeze(weight, axis=[1]))
        return reward_list, advantage, label_list, frame_is_train, weight_list

    def _calculate_single_hero_loss(self, legal_action_flag_list, unsqueeze_reward_list, unsqueeze_advantage, unsqueeze_label_list, old_label_probability_list, unsqueeze_frame_is_train, fc2_label_list, fc2_value_result, label_size_list, is_reinforce_task_list, unsqueeze_weight_list):

        reward_list, advantage, label_list, frame_is_train, weight_list = self._squeeze_tensor(
            unsqueeze_reward_list, unsqueeze_advantage, unsqueeze_label_list, unsqueeze_frame_is_train, unsqueeze_weight_list)

        train_frame_count = tf.reduce_sum(frame_is_train)
        train_frame_count = tf.maximum(
            train_frame_count, 1.0)  # prevent division by 0

        # loss of value net
        fc2_value_result_squeezed = tf.squeeze(fc2_value_result, axis=[1])
        value_cost = tf.square(
            (reward_list[0] - fc2_value_result_squeezed)) * frame_is_train
        value_cost = 0.5 * \
            tf.reduce_sum(value_cost, axis=0) / train_frame_count

        # for entropy loss calculate
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []

        # policy loss: ppo clip loss
        policy_cost = tf.constant(0.0, dtype=tf.float32)
        for task_index in range(len(is_reinforce_task_list)):
            if is_reinforce_task_list[task_index]:
                final_log_p = tf.constant(0.0, dtype=tf.float32)
                one_hot_actions = tf.one_hot(tf.to_int32(
                    label_list[task_index]), label_size_list[task_index])
                legal_action_flag_list_max_mask = (
                    1 - legal_action_flag_list[task_index]) * tf.pow(10.0, 20.0)
                label_logits_subtract_max = tf.clip_by_value((fc2_label_list[task_index] - tf.reduce_max(
                    fc2_label_list[task_index] - legal_action_flag_list_max_mask, axis=1, keep_dims=True)), -tf.pow(10.0, 20.0), 1)
                label_logits_subtract_max_list.append(
                    label_logits_subtract_max)
                label_exp_logits = legal_action_flag_list[task_index] * tf.exp(
                    label_logits_subtract_max) + self.min_policy
                label_sum_exp_logits = tf.reduce_sum(
                    label_exp_logits, axis=1, keep_dims=True)
                label_sum_exp_logits_list.append(label_sum_exp_logits)
                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)
                policy_p = tf.reduce_sum(
                    one_hot_actions * label_probability, axis=1)
                policy_log_p = tf.log(policy_p)
                #policy_log_p = tf.to_float(weight_list[task_index]) * policy_log_p
                old_policy_p = tf.reduce_sum(
                    one_hot_actions * old_label_probability_list[task_index], axis=1)
                old_policy_log_p = tf.log(old_policy_p)
                #old_policy_log_p = tf.to_float(weight_list[task_index]) * old_policy_log_p
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = tf.exp(final_log_p)
                clip_ratio = tf.clip_by_value(ratio, 0.0, 3.0)
                surr1 = clip_ratio * advantage
                surr2 = tf.clip_by_value(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                policy_cost = policy_cost - tf.reduce_sum(tf.minimum(surr1, surr2) * tf.to_float(
                    weight_list[task_index]) * frame_is_train) / tf.maximum(tf.reduce_sum(tf.to_float(weight_list[task_index]) * frame_is_train), 1.0)
                # policy_cost = - tf.reduce_sum(policy_cost, axis=0) / train_frame_count # CLIP loss, add - because need to minize

        # cross entropy loss
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(is_reinforce_task_list)):
            if is_reinforce_task_list[task_index]:
                #temp_entropy_loss = -tf.reduce_sum(label_probability_list[current_entropy_loss_index] * (label_logits_subtract_max_list[current_entropy_loss_index] - tf.log(label_sum_exp_logits_list[current_entropy_loss_index])), axis=1)
                temp_entropy_loss = -tf.reduce_sum(label_probability_list[current_entropy_loss_index] * legal_action_flag_list[task_index] * tf.log(
                    label_probability_list[current_entropy_loss_index]), axis=1)
                temp_entropy_loss = -tf.reduce_sum((temp_entropy_loss * tf.to_float(weight_list[task_index]) * frame_is_train)) / tf.maximum(
                    tf.reduce_sum(tf.to_float(weight_list[task_index]) * frame_is_train), 1.0)  # add - because need to minize
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = tf.constant(0.0, dtype=tf.float32)
                entropy_loss_list.append(temp_entropy_loss)
        entropy_cost = tf.constant(0.0, dtype=tf.float32)
        for entropy_element in entropy_loss_list:
            entropy_cost = entropy_cost + entropy_element

        # cost_all
        cost_all = value_cost + policy_cost + self.var_beta * entropy_cost

        return cost_all, value_cost, policy_cost, entropy_cost

    def _inference(self, each_hero_data_list, only_inference=True):
        whole_feature_list = State_splitter().split_features(each_hero_data_list)
        each_hero_FeatureImgLikeMg_deepfeature=[]
        for hero_index in range(len(each_hero_data_list)):
            this_hero_FeatureImgLikeMg_list = whole_feature_list[hero_index][0]
            each_hero_FeatureImgLikeMg_deepfeature.append(self.resnet_inference(
                this_hero_FeatureImgLikeMg_list, Config.resnet_FeatureImgLikeMg_n, reuse=tf.compat.v1.AUTO_REUSE))

        hero_fc1_weight_list = []
        hero_fc1_bias_list = []
        hero_fc1_weight_add_list = []
        hero_fc1_bias_add_list = []
        hero_fc2_weight_list = []
        hero_fc2_bias_list = []
        feature_data_list = []
        for hero_index in range(len(each_hero_data_list)):
            feature_data_list.append(tf.identity(
                each_hero_data_list[hero_index][0], name="hero%d_feature" % (hero_index)))
            hero_fc1_weight_list.append(self._fc_weight_variable(
                shape=[4586, 64], name="fc1_hero%d_main_weight" % (hero_index)))
            hero_fc1_bias_list.append(self._bias_variable(
                shape=[64], name="fc1_hero%d_main_bias" % (hero_index)))

            hero_fc1_weight_add_list.append(self._fc_weight_variable(
                shape=[64, 64], name="fc1_hero%d_main_weight_add" % (hero_index)))
            hero_fc1_bias_add_list.append(self._bias_variable(
                shape=[64], name="fc1_hero%d_main_bias_add" % (hero_index)))

            hero_fc2_weight_list.append(self._fc_weight_variable(
                shape=[64, 64], name="fc2_hero%d_main_weight" % (hero_index)))
            hero_fc2_bias_list.append(self._bias_variable(
                shape=[64], name="fc2_hero%d_main_bias" % (hero_index)))

        hero_fc1_result_list = []
        hero_fc1_result_add_list = []
        hero_fc2_result_list = []
        for hero_index in range(len(each_hero_data_list)):
            hero_fc1_result_list.append(tf.nn.relu((tf.matmul(
                feature_data_list[hero_index], hero_fc1_weight_list[hero_index]) + hero_fc1_bias_list[hero_index]), name="hero%d_fc1_result" % (hero_index)))
            hero_fc1_result_add_list.append(tf.nn.relu((tf.matmul(
                hero_fc1_result_list[hero_index], hero_fc1_weight_add_list[hero_index]) + hero_fc1_bias_add_list[hero_index]), name="hero%d_fc1_result_add" % (hero_index)))
            hero_fc2_result_list.append(tf.nn.relu((tf.matmul(
                hero_fc1_result_add_list[hero_index], hero_fc2_weight_list[hero_index]) + hero_fc2_bias_list[hero_index]), name="hero%d_fc2_result" % (hero_index)))

        hero_fc1_label_weight_list = []
        hero_fc1_label_bias_list = []
        hero_fc1_value_weight_list = []
        hero_fc1_value_bias_list = []
        for hero_index in range(len(each_hero_data_list)):
            fc1_label_weight_list = []
            fc1_label_bias_list = []
            for label_index in range(len(self.hero_label_size_list[hero_index])):
                fc1_label_weight_list.append(self._fc_weight_variable(
                    shape=[64, self.hero_label_size_list[hero_index][label_index]], name="hero%d_fc1_label_%d_weight" % (hero_index, label_index)))
                fc1_label_bias_list.append(self._bias_variable(
                    shape=[self.hero_label_size_list[hero_index][label_index]], name="hero%d_fc1_label_%d_bias" % (hero_index, label_index)))

            hero_fc1_label_weight_list.append(fc1_label_weight_list)
            hero_fc1_label_bias_list.append(fc1_label_bias_list)
            # value fc para
            hero_fc1_value_weight_list.append(self._fc_weight_variable(
                shape=[64, 1], name="hero%d_fc1_value_weight" % (hero_index)))
            hero_fc1_value_bias_list.append(self._bias_variable(
                shape=[1], name="hero%d_fc1_value_bias" % (hero_index)))

        each_hero_fc_result_list = []
        for hero_index in range(len(each_hero_data_list)):
            this_hero_fc_result_list = []
            this_hero_variable_list = []
            for label_index in range(len(self.hero_label_size_list[hero_index])):
                this_hero_fc_result_list.append(tf.add(tf.matmul(hero_fc2_result_list[hero_index], hero_fc1_label_weight_list[hero_index][
                                                label_index]), hero_fc1_label_bias_list[hero_index][label_index], name="hero%d_fc1_label_%d_result" % (hero_index, label_index)))

            this_hero_fc_result_list.append(tf.add(tf.matmul(
                hero_fc2_result_list[hero_index], hero_fc1_value_weight_list[hero_index]), hero_fc1_value_bias_list[hero_index], name="hero%d_fc1_value_result" % hero_index))

            each_hero_fc_result_list.append(this_hero_fc_result_list)

        return each_hero_fc_result_list

    def _conv_weight_variable(self, shape, name, trainable=True):
        #initializer = tf.contrib.layers.xavier_initializer_conv2d()
        initializer = tf.orthogonal_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)

    def _fc_weight_variable(self, shape, name, trainable=True):
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.orthogonal_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)

    def _bias_variable(self, shape, name, trainable=True):
        initializer = tf.constant_initializer(0.0)
        return tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)

    def _checkpoint_filename(self, episode):
        return './checkpoints/'

    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])


class State_splitter():
    '''
    split states of 3 heros into kaiwu website format, use function split_features
    '''
    def  __init__(self):
        pass   

    def _split_features_one_hero(self, this_hero_data_list):
        '''
        split features of one hero into kaiwu website format
        '''
        # this_hero_data_list = each_hero_data_list[0]
        # get the image like feature 1d vector arranged in (6,17,17) order, and convert that into tensor of shape (height, width, depth) (1,17,17,6)
        this_hero_FeatureImgLikeMg_vec_list = tf.reshape(this_hero_data_list[:6*17*17],[6, 17, 17])
        this_hero_FeatureImgLikeMg_list = tf.transpose(
            this_hero_FeatureImgLikeMg_vec_list, perm=[1, 2, 0])
        # get VecFeatureHero and split it into shape of (1,6,251) [us 3, opponent 3]
        this_hero_VecFeatureHero_list = tf.reshape(this_hero_data_list[1734:3239+1],[6, 251])
        # get MainHeroFeature with shape of (1,44)
        this_hero_MainHeroFeature_list = this_hero_data_list[3240:3283+1]
        # get VecSoldier and split it into shape of (1,20,25)
        this_hero_VecSoldier_list = tf.reshape(this_hero_data_list[3284:3783+1],[20, 25])
        # get VecOrgan and split it into shape of (1,6,29)
        this_hero_VecOrgan_list = tf.reshape(this_hero_data_list[3784:3957+1],[6, 29])
        # get VecMonster and split it into shape of (1,20,28)
        this_hero_VecMonster_list = tf.reshape(this_hero_data_list[3958:4517+1],[20, 28])
        # get VecCampsWholeInfo with shape of (1,68)
        this_hero_VecCampsWholeInfo_list = this_hero_data_list[4518:4585+1]
        whole_feature_list=[]
        for items in [this_hero_FeatureImgLikeMg_list, this_hero_VecFeatureHero_list, this_hero_MainHeroFeature_list, this_hero_VecSoldier_list, this_hero_VecOrgan_list, this_hero_VecMonster_list, this_hero_VecCampsWholeInfo_list]:
            whole_feature_list.append(tf.expand_dims(items,axis=0))
        return whole_feature_list

    def split_features(self, each_hero_data_list):
        splitted_features = []
        for hero_index in range(len(each_hero_data_list)):
            splitted_features.append(self._split_features_one_hero(
                tf.squeeze(each_hero_data_list[hero_index][0])))
        return splitted_features
    
    def get_states_names(self):
        ['FeatureImgLikeMg','VecFeatureHero','MainHeroFeature','VecSoldier','VecOrgan','VecMonster','VecCampsWholeInfo']
    

class Feature_extraction():
    '''extract features. from raw state data to deep features'''

    def __init__(self):
        pass

    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False, trainable=True):
        '''
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: User Xavier as default.
        :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
        layers.
        :return: The created variable
        '''

        # TODO: to allow different weight decay to fully connected layer and conv layer
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=tf.constant(0,dtype=tf.float32))

        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                        regularizer=regularizer, trainable=trainable)
        return new_variables

    def output_layer(self, input_layer, num_labels):
        '''
        (Not used)
        :param input_layer: 2D tensor
        :param num_labels: int. How many output labels in total? Not used
        :return: output layer Y = WX + B
        '''
        input_dim = input_layer.get_shape().as_list()[-1]
        fc_w = self.create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = self.create_variables(name='fc_bias', shape=[
                                num_labels], initializer=tf.zeros_initializer())

        fc_h = tf.matmul(input_layer, fc_w) + fc_b
        return fc_h


    def fc_layer(self, input_vec, output_dims):
        '''
        basic fully connected layer
        :param input_vec: input state vector
        :param output_dims: output dimensions
        :return: output layer Y = WX + B
        '''
        input_dims = input_vec.get_shape().as_list()[-1]
        fc_w = self.create_variables(name='fc_weights', shape=[input_dims, output_dims], is_fc_layer=True,
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = self.create_variables(name='fc_bias', shape=[
                                output_dims], initializer=tf.zeros_initializer())

        fc_h = tf.matmul(input_vec, fc_w) + fc_b
        return fc_h
    
    def bn_relu_fc_layer(self, input_layer, output_dims):
        '''
        A helper function to  batch normalize, relu and fc the input tensor sequentially
        :param input_layer: 1D tensor
        :param output_dims: int. 
        :return: 1D tensor. Y = Relu(batch_normalize(conv(X)))
        '''
        
        
        #bn part
        mean, variance = tf.nn.moments(input_layer, axes=[0])
        beta = tf.get_variable('beta', 1, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', 1, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(
            input_layer, mean, variance, beta, gamma, Config.BN_EPSILON)

        relu_layer = tf.nn.relu(bn_layer)
        fc_layer = self.fc_layer(relu_layer, output_dims)
        return output
    
    def fc_bn_relu_layer(self, input_layer, output_dims):
        '''
        A helper function to  fc, batch normalize, relu  the input tensor sequentially
        :param input_layer: 1D tensor
        :param output_dims: int. 
        :return: 1D tensor. Y = Relu(batch_normalize(fc(X)))
        '''
        
        
        #bn part
        fc_layer = self.fc_layer(input_layer, output_dims)
        mean, variance = tf.nn.moments(fc_layer, axes=[0])
        beta = tf.get_variable('beta', 1, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', 1, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(
            fc_layer, mean, variance, beta, gamma, Config.BN_EPSILON)

        relu_layer = tf.nn.relu(bn_layer)
        
        return output
    
    def res_fc_block(self, input_layer, output_dims):
        '''
        res_fc_block (unfinished)
        '''

        input_dim = input_layer.get_shape().as_list()[-1]
        if not input_dim==output_dims:
            if_change_dim = True  
        else:
            if_change_dim = False
        fc_layer=bn_relu_fc_layer(self, input_layer, output_dims)
        
        ksize_pool=input_dim//output_dims+1
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[ksize_pool],
                                          strides=[ksize_pool], padding='SAME')
            # padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
            #                                                               input_channel // 2]])
        return padded_input


    def batch_normalization_layer(self, input_layer, dimension):
        '''
        Helper function to do batch normalziation
        :param input_layer: 4D tensor
        :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
        :return: the 4D tensor after being normalized
        '''
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(
            input_layer, mean, variance, beta, gamma, Config.BN_EPSILON)

        return bn_layer

    def conv_bn_relu_layer(self, input_layer, filter_shape, stride):
        '''
        A helper function to conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        '''

        out_channel = filter_shape[-1]
        filter = self.create_variables(name='conv', shape=filter_shape)

        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[
                                  1, stride, stride, 1], padding='SAME')
        bn_layer = self.batch_normalization_layer(conv_layer, out_channel)

        output = tf.nn.relu(bn_layer)
        return output

    def bn_relu_conv_layer(self, input_layer, filter_shape, stride):
        '''
        A helper function to batch normalize, relu and conv the input layer sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        '''

        in_channel = input_layer.get_shape().as_list()[-1]

        bn_layer = self.batch_normalization_layer(input_layer, in_channel)
        relu_layer = tf.nn.relu(bn_layer)

        filter = self.create_variables(name='conv', shape=filter_shape)
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[
                                  1, stride, stride, 1], padding='SAME')
        return conv_layer

    def residual_block(self, input_layer, output_channel, first_block=False):
        '''
        Defines a residual block in ResNet
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        '''
        input_channel = input_layer.get_shape().as_list()[-1]

        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError(
                'Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = self.create_variables(
                    name='conv', shape=[3, 3, input_channel, output_channel])
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[
                                     1, 1, 1, 1], padding='SAME')
            else:
                conv1 = self.bn_relu_conv_layer(
                    input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = self.bn_relu_conv_layer(
                conv1, [3, 3, output_channel, output_channel], 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                          input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output

    def resnet_inference(self, input_tensor_batch, n, reuse):
        '''
        The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
        :param input_tensor_batch: 4D tensor [batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH]; conv filter [filter_height, filter_width, filter_depth, filter_number]
        :param n: num_residual_blocks
        :param reuse: To build train graph, reuse=False. To build validation graph and share weights
        with train graph, resue=True
        :return: last layer in the network. Not softmax-ed
        '''

        layers = []
        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = self.conv_bn_relu_layer(input_tensor_batch, [7, 7, 6, 16], 1)
            # activation_summary(conv0)
            layers.append(conv0)

        for i in range(n):
            with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                if i == 0:
                    conv1 = self.residual_block(layers[-1], 16, first_block=True)
                else:
                    conv1 = self.residual_block(layers[-1], 16)
                # activation_summary(conv1)
                layers.append(conv1)

        for i in range(n):
            with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                conv2 = self.residual_block(layers[-1], 32)
                # activation_summary(conv2)
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                conv3 = self.residual_block(layers[-1], 64)
                layers.append(conv3)
            assert conv3.get_shape().as_list()[1:] == [5, 5, 64]

        with tf.variable_scope('fc', reuse=reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = self.batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            assert global_pool.get_shape().as_list()[-1:] == [64]
            # output = self.output_layer(global_pool, 10)
            layers.append(global_pool)

        return layers[-1]

    def vec_feature_extraction(self,vec_state_list):
        input_dim_list=[vec_state.get_shape().as_list()[-1] for vec_state in vec_state_list]
        # num of units per class of states e.g.10 for VecSoldier since there are 10 soldiers
        num_unit_perState_list=[vec_state.get_shape().as_list()[-2] for vec_state in vec_state_list] 
        state_name_list=State_splitter.get_states_names()[1:]

        assert len(state_name_list)==len(input_dim_list)

        output_vec_feature_list=[]
        for i,state_name in enumerate(state_name_list):
            this_unit_feature_list=[]
            for j in range(num_unit_perState_list[i])
                with tf.variable_scope(state_name+f'_{j}', reuse=reuse):
                    input_layer=vec_state_list[i][j]
                    this_unit_feature_list.append(fc_bn_relu_layer(self, input_layer,Config.vec_feat_extract_out_dims[i]))
            output_vec_feature_list.append(this_unit_feature_list)
        
        return output_vec_feature_list
            

    def img_feature_extraction(self, input_tensor_batch, n, reuse):
        '''
        Main function for imglike feature extraction. input shape[1,17,17,6], output shape[64]
        :param: input_tensor_batch: input img like feature. Shape [1,17,17,6]([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        :param: n: num_residual_blocks
        '''
        return self.resnet_inference(self, input_tensor_batch, n, reuse)
