#__author__ = "yannickyu"

import tensorflow as tf
import glob
import numpy as np
import h5py
from config.Config import Config
import sys


class NetworkModel():

    def __init__(self, batch_size=0, use_xla=False):
        # feature configure parameter
        self.model_name = Config.NETWORK_NAME
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
        self.restore_list = []
        self.min_policy = Config.MIN_POLICY
        #self.batch_size = batch_size
        self.embedding_trainable = False
        self.use_xla = use_xla

        self.hero_num = 3
        self.hero_data_len = sum(Config.data_shapes[0])
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

    def build_graph(self, datas, global_step=None):
        datas = tf.reshape(datas, [-1, self.hero_num, self.hero_data_len])
        data_list = tf.transpose(datas, perm=[1, 0, 2])

        # new add 20180912
        each_hero_data_list = []
        each_hero_lstm_cell = []
        each_hero_lstm_hidden = []
        for hero_index in range(self.hero_num):
            this_hero_data_list, this_hero_lstm_cell, this_hero_lstm_hidden = self._split_data(
                tf.cast(data_list[hero_index], tf.float32), self.hero_data_split_shape[hero_index], hero_index)
            each_hero_data_list.append(this_hero_data_list)
            each_hero_lstm_cell.append(this_hero_lstm_cell)
            each_hero_lstm_hidden.append(this_hero_lstm_hidden)

        self.lstm_cell_ah = each_hero_lstm_cell
        self.lstm_hidden_ah = each_hero_lstm_hidden
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

    def _split_data(self, this_hero_data, this_hero_data_split_shape, hero_id):
        # calculate length of each frame
        this_hero_each_frame_data_length = np.sum(
            np.array(this_hero_data_split_shape))
        # TODO: LSTM unit size should increase
        # TODO: should have lstm_time_steps numbner of LSTMcell?
        this_hero_sequence_data_length = this_hero_each_frame_data_length * self.lstm_time_steps
        this_hero_sequence_data_split_shape = np.array(
            [this_hero_sequence_data_length, self.lstm_unit_size, self.lstm_unit_size])
        sequence_data, lstm_cell_data_ah, lstm_hidden_data_ah = tf.split(
            this_hero_data, this_hero_sequence_data_split_shape, axis=1)
        lstm_cell_data=tf.split(lstm_cell_data_ah, num_or_size_splits=3, axis=1)[hero_id]
        lstm_hidden_data=tf.split(lstm_hidden_data_ah, num_or_size_splits=3, axis=1)[hero_id]
        reshape_sequence_data = tf.reshape(
            sequence_data, [-1, self.lstm_time_steps, this_hero_each_frame_data_length])
        sequence_this_hero_data_list = tf.split(
            reshape_sequence_data, np.array(this_hero_data_split_shape), axis=2)
        # self.lstm_cell_ph = lstm_cell_data
        # self.lstm_hidden_ph = lstm_hidden_data
        return sequence_this_hero_data_list, lstm_cell_data, lstm_hidden_data

    def _get_last_tstep_data(self, each_hero_data_list_allt):
        '''only last time step data is used for calculation. Not sure if this is correct'''
        each_hero_data_list = []
        for i in range(len(each_hero_data_list_allt)):
            this_hero_data_list = []
            for data in each_hero_data_list_allt[i]:
                this_hero_data_list.append(data[:, -1, :])
            each_hero_data_list.append(this_hero_data_list)
        return each_hero_data_list

    def _calculate_loss(self, each_hero_data_list_allt, each_hero_fc_result_list):
        each_hero_data_list = self._get_last_tstep_data(
            each_hero_data_list_allt)
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
        # split states
        lstm_cell_all_hero = self.lstm_cell_ah
        lstm_hidden_all_hero = self.lstm_hidden_ah
        whole_feature_list = State_splitter().split_features(each_hero_data_list)
        # feature extraction
        extracted_feature = Feature_extraction().get_extracted_feature(whole_feature_list)
        # LSTM, hidden_out is output
        # feature_dim=extracted_feature[0].get_shape().as_list()[-1]
        lstm_module=LSTM(lstm_hidden_dim=self.lstm_hidden_dim/3) #hidden_dim=512 per hero
        cell_out, hidden_out = lstm_module.lstm_inference(
            extracted_feature, lstm_cell_all_hero, lstm_hidden_all_hero)
        self.lstm_cell_output = lstm_module.reshape_for_lstm_output(cell_out)
        self.lstm_hidden_output = lstm_module.reshape_for_lstm_output(hidden_out)
        # Communications
        Comm_out = Communication().COMM_inference(hidden_out)
        each_hero_fc_result_list = ActionChooser(Config=Config).Action_inference(Comm_out)

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

    def __init__(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS

    def _split_features_one_hero(self, this_hero_data_list):
        '''
        split features of one hero into kaiwu website format
        '''
        tstpes = self.lstm_time_steps
        # this_hero_data_list = each_hero_data_list[0]
        # get the image like feature 1d vector arranged in (6,17,17) order, and convert that into tensor of shape (height, width, depth) (1,17,17,6)
        this_hero_FeatureImgLikeMg_vec_list = tf.reshape(
            this_hero_data_list[:, :, :6*17*17], [-1, tstpes, 6, 17, 17])
        this_hero_FeatureImgLikeMg_list = tf.transpose(
            this_hero_FeatureImgLikeMg_vec_list, perm=[0, 1, 3, 4, 2])
        # get VecFeatureHero and split it into shape of (1,6,251) [us 3, opponent 3]
        this_hero_VecFeatureHero_list = tf.reshape(
            this_hero_data_list[:, :, 1734:3239+1], [-1, tstpes, 6, 251])
        # get MainHeroFeature with shape of (1,1,44)
        this_hero_MainHeroFeature_list = tf.reshape(
            this_hero_data_list[:, :, 3240:3283+1], [-1, tstpes, 1, 44])
        # get VecSoldier and split it into shape of (1,20,25)
        this_hero_VecSoldier_list = tf.reshape(
            this_hero_data_list[:, :, 3284:3783+1], [-1, tstpes, 20, 25])
        # get VecOrgan and split it into shape of (1,6,29)
        this_hero_VecOrgan_list = tf.reshape(
            this_hero_data_list[:, :, 3784:3957+1], [-1, tstpes, 6, 29])
        # get VecMonster and split it into shape of (1,20,28)
        this_hero_VecMonster_list = tf.reshape(
            this_hero_data_list[:, :, 3958:4517+1], [-1, tstpes, 20, 28])
        # get VecCampsWholeInfo with shape of (1,1,68)
        this_hero_VecCampsWholeInfo_list = tf.reshape(
            this_hero_data_list[:, :, 4518:4585+1], [-1, tstpes, 1, 68])
        whole_feature_list = []
        for items in [this_hero_FeatureImgLikeMg_list, this_hero_VecFeatureHero_list, this_hero_MainHeroFeature_list, this_hero_VecSoldier_list, this_hero_VecOrgan_list, this_hero_VecMonster_list, this_hero_VecCampsWholeInfo_list]:
            whole_feature_list.append(items)
        return whole_feature_list

    def split_features(self, each_hero_data_list):
        splitted_features = []
        for hero_index in range(len(each_hero_data_list)):
            splitted_features.append(self._split_features_one_hero(
                each_hero_data_list[hero_index][0]))
        return splitted_features

    def get_states_names(self):
        ['FeatureImgLikeMg', 'VecFeatureHero', 'MainHeroFeature',
            'VecSoldier', 'VecOrgan', 'VecMonster', 'VecCampsWholeInfo']

    def merge_tsteps_dim_to_batch(self, splitted_features):
        '''
        each elements of splitted_features, turn shape [?,tsteps,x,y] to [?,x,y]
        where tsteps is listm_time_steps. For all heros.
        '''

        merged_features_ah = []
        for i in range(len(splitted_features)):
            merged_features = []
            for f in splitted_features[i]:
                original_shape = f.get_shape().as_list()
                merged_shape = [-1]+original_shape[2:]
                mf = tf.reshape(f, merged_shape)
                merged_features.append(mf)
            merged_features_ah.append(merged_features)
        return merged_features_ah

    def detach_tsteps_dim_from_batch(self, merged_features):
        '''
        inverse of def merge_tsteps_dim_to_batch(self,splitted_features). For all heros.
        '''
        de_features_ah = []
        for i in range(len(merged_features)):
            de_features = []
            for f in merged_features[i]:
                original_shape = f.get_shape().as_list()
                de_shape = [-1]+[self.lstm_time_steps]+original_shape[1:]
                df = tf.reshape(f, de_shape)
                de_features.append(df)
            de_features_ah.append(de_features)
        return de_features_ah


class Feature_extraction():
    '''extract features. from raw state data to deep features'''

    def __init__(self):
        self.reuse = Config.reuse

    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True):
        '''
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: User Xavier as default.
        :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
        layers.
        :return: The created variable
        '''

        # TODO: to allow different weight decay to fully connected layer and conv layer
        # regularizer = tf.contrib.layers.l2_regularizer(
        #     scale=tf.constant(0, dtype=tf.float32))

        new_variables = tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)
        return new_variables

    

    def fc_layer(self, input_vec, output_dims, fc_name):
        '''
        basic fully connected layer
        :param input_vec: input state vector
        :param output_dims: output dimensions
        :return: output layer Y = WX + B
        '''
        input_dims = input_vec.get_shape().as_list()[-1]
        input_dims_num = len(input_vec.get_shape().as_list())
        fc_w = self.create_variables(name=fc_name+'_fc_weights', shape=[input_dims, output_dims], 
                                     initializer=Config.vecNet_fc_weight_initializer)
        fc_b = self.create_variables(name=fc_name+'_fc_bias', shape=[
            output_dims], initializer=Config.vecNet_fc_bias_initializer)
        # if len(input_vec.get_shape().as_list())==1:
        # input_vec = tf.expand_dims(input_vec, 0)
        fc_h = tf.matmul(input_vec, fc_w) + fc_b
        return fc_h

    def bn_relu_fc_layer(self, input_layer, output_dims, if_block, input_block_w):
        '''
        A helper function to  batch normalize, relu and fc the input tensor sequentially
        :param input_layer: 1D tensor
        :param output_dims: int. 
        :return: 1D tensor. Y = Relu(batch_normalize(conv(X)))
        '''

        # bn part
        dimensions = input_layer.get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(input_layer, axes=[0])
        beta = tf.get_variable('beta', dimensions, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimensions, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(
            input_layer, mean, variance, beta, gamma, Config.BN_EPSILON)

        relu_layer = tf.nn.relu(bn_layer)
        if if_block:
            output = tf.matmul(input_layer, input_block_w)
        else:
            output = self.fc_layer(relu_layer, output_dims)
        return output

    def fc_bn_relu_layer(self, input_layer, output_dims, if_bn, if_block,  fc_relu_name, input_block_w=None):
        '''
        A helper function to  fc, batch normalize, relu  the input tensor sequentially
        :param input_layer: 1D tensor
        :param output_dims: int. 
        :param if_bn: if BN layer is used
        :param if_block: if vec net is in the form of block mat mul
        :return: 1D tensor. Y = Relu(batch_normalize(fc(X)))
        '''

        if if_block:
            fc_layer = tf.matmul(input_layer, input_block_w)
        else:
            fc_layer_result = self.fc_layer(input_layer, output_dims, fc_name=fc_relu_name)
        # bn part
        dimensions = fc_layer_result.get_shape().as_list()[-1]
        if if_bn:
            mean, variance = tf.nn.moments(fc_layer_result, axes=[0])
            beta = tf.get_variable('beta', dimensions, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', dimensions, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
            bn_layer = tf.nn.batch_normalization(
                fc_layer_result, mean, variance, beta, gamma, Config.BN_EPSILON)

            relu_layer = tf.nn.relu(bn_layer)
        else:
            relu_layer = tf.nn.relu(fc_layer_result)

        return relu_layer

    def res_fc_block(self, input_layer, output_dims, if_block, input_block_w, num_vec_fc_in_resblock=Config.num_vec_fc_in_resblock):
        '''
        res_fc_block
        '''

        input_dim = input_layer.get_shape().as_list()[-1]
        if not input_dim == output_dims:
            if_change_dim = True
        else:
            if_change_dim = False

        with tf.variable_scope('fc1_in_resblock'):
            fc_layer1 = self.bn_relu_fc_layer(
                input_layer, output_dims, if_block, input_block_w)
            fc_layer_out = fc_layer1
        if num_vec_fc_in_resblock == 2:
            with tf.variable_scope('fc2_in_resblock'):
                fc_layer2 = self.bn_relu_fc_layer(
                    fc_layer1, output_dims, if_block, input_block_w)
                fc_layer_out = fc_layer2
        # if not is_change_dim is True:
        resoutput = input_layer+fc_layer_out

        # pad_dim = (input_dim-output_dims)
        # if if_change_dim is True:
        #     # pooled_input = tf.nn.avg_pool(input_layer, ksize=[ksize_pool],
        #     #                               strides=[ksize_pool], padding='SAME')
        #     pad_input = tf.pad(input_layer, [[0,pad_dim]])
        #     resoutput=pad_input+
        return resoutput

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

    def conv_bn_relu_layer(self, input_layer, filter_shape, stride,layer_name):
        '''
        A helper function to conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        bN is not used!
        '''

        out_channel = filter_shape[-1]
        kel_filter = self.create_variables(name=layer_name+'_kernel', shape=filter_shape,initializer=Config.img_conv_initializer)

        conv_layer = tf.nn.conv2d(input_layer, kel_filter, strides=[
                                  1, stride, stride, 1], padding='SAME')
        # bn_layer = self.batch_normalization_layer(conv_layer, out_channel)

        output = tf.nn.relu(conv_layer)
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

    def conv_2_inference(self, input_tensor_batch,hero_id_for_name):
        # conv_bn_relu_layer(self, input_layer, filter_shape, stride): filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        layers = []
        conv0 = self.conv_bn_relu_layer(
            input_tensor_batch, [7, 7, 6, 16], 1,layer_name='hero'+f'{hero_id_for_name}'+'_imgconv_conv0')
        layers.append(conv0)
        pool_layer = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[
                                    1, 2, 2, 1], padding='SAME',name='hero'+f'{hero_id_for_name}'+'_imgconv_maxpool')
        layers.append(pool_layer)
        conv1 = self.conv_bn_relu_layer(
            pool_layer, [3, 3, 16, 16], 2,layer_name='hero'+f'{hero_id_for_name}'+'_imgconv_conv1')
        layers.append(conv1)
        flat_dim = np.prod(layers[-1].get_shape().as_list()[1:])
        outlayer = tf.reshape(layers[-1], [-1, flat_dim])
        return outlayer

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
        with tf.variable_scope('imgres_conv0', reuse=self.reuse):
            conv0 = self.conv_bn_relu_layer(
                input_tensor_batch, [7, 7, 6, 16], 1)
            # activation_summary(conv0)
            layers.append(conv0)

        for i in range(n):
            with tf.variable_scope('imgres_conv1_%d' % i, reuse=self.reuse):
                if i == 0:
                    conv1 = self.residual_block(
                        layers[-1], 16, first_block=True)
                else:
                    conv1 = self.residual_block(layers[-1], 16)
                # activation_summary(conv1)
                layers.append(conv1)

        for i in range(n):
            with tf.variable_scope('imgres_conv2_%d' % i, reuse=self.reuse):
                conv2 = self.residual_block(layers[-1], 32)
                # activation_summary(conv2)
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('imgres_conv3_%d' % i, reuse=self.reuse):
                conv3 = self.residual_block(layers[-1], 64)
                layers.append(conv3)
            assert conv3.get_shape().as_list()[1:] == [5, 5, 64]

        with tf.variable_scope('fc', reuse=self.reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = self.batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            assert global_pool.get_shape().as_list()[-1:] == [64]
            # output = self.output_layer(global_pool, 10)
            layers.append(global_pool)

        return layers[-1]

    def vec_fc_first_layer(self, input_layer, output_dims, if_bn, if_block,  fc_first_name):
        return self.fc_bn_relu_layer(input_layer, output_dims, if_bn, if_block, fc_relu_name=fc_first_name)

    def vec_fc_second_layer(self, input_layer, output_dims, vec_fc_2ndlayer_type, if_bn, if_block, fc_second_name):
        if vec_fc_2ndlayer_type == 'resfc':
            output_layer = self.res_fc_block(
                input_layer, output_dims, if_block, input_block_w)
        if vec_fc_2ndlayer_type == 'fc':
            output_layer = self.fc_bn_relu_layer(
                input_layer, output_dims, if_bn, if_block,fc_relu_name=fc_second_name)
        return output_layer
    
    def vec_fc_third_layer(self, input_vec, output_dims, fc_name):
        return self.fc_layer(input_vec, output_dims, fc_name)

    def vec_feature_extraction(self, vec_state_list, hero_id_for_name, if_quicker_implementation=False):
        if_block = if_quicker_implementation
        input_dim_list = [vec_state.get_shape().as_list()[-1]
                          for vec_state in vec_state_list]
        # num of units per class of states e.g.10 for VecSoldier since there are 10 soldiers
        num_unit_perState_list = [vec_state.get_shape(
        ).as_list()[-2] for vec_state in vec_state_list]
        state_name_list = Config.states_names[1:]

        assert len(state_name_list) == len(input_dim_list)
        output_vec_feature_list = []
        for i, state_name in enumerate(state_name_list):
            this_unit_feature_list = []
            for j in range(num_unit_perState_list[i]):
                input_layer = vec_state_list[i][:, j, :]
                output_dims_fc1 = Config.vec_feat_extract_out_dims[0][i]
                output_dims_fc2 = Config.vec_feat_extract_out_dims[1][i]
                output_dims_fc3 = Config.vec_feat_extract_out_dims[2][i]
                vec_name_pre_now='hero'+f'{hero_id_for_name}'+'_'+state_name+'_unit'+f'{j}'
                if state_name=='VecCampsWholeInfo':#for VecCampsWholeInfo, only one fc
                    fc3=self.vec_fc_third_layer(input_layer, output_dims_fc3, fc_name=vec_name_pre_now+'vec_fc_3')
                else:
                    #1,fc-relu
                    fc1 = self.vec_fc_first_layer(
                        input_layer, output_dims_fc1, Config.if_vec_fc_bn, if_block, fc_first_name=vec_name_pre_now+'_vec_fc_1')
                    #2,fc-relu
                    fc2 = self.vec_fc_second_layer(
                        fc1, output_dims_fc2, Config.vec_fc_2ndlayer_type, Config.if_vec_fc_bn, if_block,fc_second_name=vec_name_pre_now+'vec_fc_2')
                    #3,fc      
                    fc3=self.vec_fc_third_layer(fc2, output_dims_fc3, fc_name=vec_name_pre_now+'vec_fc_3')
                this_unit_feature_list.append(fc3)

            output_vec_feature_list.append(this_unit_feature_list)
        
        return output_vec_feature_list

    def _get_block_matrix(self, blocks):
        linop_blocks = [tf.linalg.LinearOperatorFullMatrix(
            block) for block in blocks]
        linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)

    def _maxpooling_in_state_group(self,output_vec_feature_list):#last layer of FE. Do max pooling between units of same side
        result_feat_list=[]
        for i,vec_units in enumerate(output_vec_feature_list):
            split_index=Config.vec_unit_group_split_id[i]
            if split_index is not None:
                group_frd=vec_units[:split_index]
                group_enemy=vec_units[split_index:]
                stacked_frd=tf.stack(group_frd,axis=0)
                stacked_enemy=tf.stack(group_enemy,axis=0)
                pooled_frd=tf.compat.v2.reduce_max(stacked_frd,axis=0)
                pooled_enemy=tf.compat.v2.reduce_max(stacked_enemy,axis=0)
                result_feat_list.append(pooled_frd)
                result_feat_list.append(pooled_enemy)
            else:
                result_feat_list.append(vec_units[0])
        return result_feat_list


    def img_feature_extraction(self, input_tensor_batch, n, img_net_type,hero_id_for_name):
        '''
        Main function for imglike feature extraction. input shape[1,17,17,6], output shape[64]
        :param: input_tensor_batch: input img like feature. Shape [1,17,17,6]([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        :param: n: num_residual_blocks
        '''
        if img_net_type == 'img_res':
            output = self.resnet_inference(input_tensor_batch, n, reuse=None)
        if img_net_type == 'img_conv':
            output = self.conv_2_inference(input_tensor_batch,hero_id_for_name)
        return output

    def get_extracted_feature(self, whole_feature_list_unmerged):
        whole_feature_list = State_splitter().merge_tsteps_dim_to_batch(
            whole_feature_list_unmerged)
        extracted_feature_all_heros = []
        for i, each_hero_feature in enumerate(whole_feature_list):
            # with tf.variable_scope('hero'+f'_{i}', reuse=self.reuse):
            n = Config.resnet_FeatureImgLikeMg_n
            img_feature_extracted = [self.img_feature_extraction(
                each_hero_feature[0], Config.img_num_res_blocks,  Config.img_net_type, hero_id_for_name=i)]
            vec_feature_extracted_list = self.vec_feature_extraction(
                each_hero_feature[1:],hero_id_for_name=i)
            vec_feature_maxpool_list=self._maxpooling_in_state_group(vec_feature_extracted_list)
            # vec_feature_extracted_list_flatten = []
            # for vec in vec_feature_maxpool_list:
            #     vec_feature_extracted_list_flatten.append(
            #         tf.concat(vec, 1))
            whole_feature_list_extracted = img_feature_extracted + \
                vec_feature_maxpool_list
            whole_feature_extracted = tf.concat(
                whole_feature_list_extracted, 1)
            extracted_feature_all_heros.append(whole_feature_extracted)
        return extracted_feature_all_heros


class LSTM():
    def __init__(self, lstm_hidden_dim):
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_cell_ah = [tf.nn.rnn_cell.LSTMCell(
            self.lstm_hidden_dim) for _ in range(3)]
        self.reuse = Config.reuse
        self.tstep = Config.LSTM_TIME_STEPS
        # self.lstm_multi_cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell]*self.tstep)
    # LSTM Start
    # def _init_lstm_cell(self):
    #     # lstm_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_dim)
    #     lstm_cell = tf.keras.layers.LSTM(self.lstm_hidden_dim)
    #     return lstm_cell

    def _lstm_cell_forward(self, lstm_input, cell_ph, hidden_ph):
        # lstm_input: [batch_size, input_size]
        # cell_ph: [batch_size, self.lstm_hidden_dim]
        # hidden_ph: [batch_size, self.lstm_hidden_dim]
        hidden_out, out = self.lstm_cell(lstm_input, [cell_ph, hidden_ph])
        cell_out = out[0]

        return cell_out, hidden_out

    def _lstm_multi_cell_forward(self, lstm_cell, lstm_input, cell_ph, hidden_ph):
        '''
        :param lstm_input: lstm input per hero with shape [batch_size, tstps, input_dim]
        :param cell_ph: cell input with shape [batch_size, LSTM_UNIT_SIZE]
        :param hidden_ph: hidden cell input with shape [batch_size, LSTM_UNIT_SIZE]
        '''
        # hidden_out_list = []
        # out_list = []
        # for t in range(self.tstep):
        #     hidden, state=self.lstm_multi_cell(lstm_input[:,t,:], [cell_ph, hidden_ph])
        #     hidden_list.append(hidden)
        #     state_list.append(state)
        # hidden_out=hidden_list[-1]
        # cell_out=state_list[-1,0]
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, lstm_input, initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                                                 cell_ph, hidden_ph), time_major=False)
        cell_out = final_state[0]
        hidden_out = final_state[1]
        return cell_out, hidden_out

    def _detach_tsteps(self, extracted_feature):
        '''
        detach tsteps dim for one hero
        '''
        original_shape = extracted_feature.get_shape().as_list()
        de_shape = [-1]+[self.tstep]+original_shape[1:]
        df = tf.reshape(extracted_feature, de_shape)
        return df
    
    def reshape_for_lstm_output(self,state_list):
        return tf.concat(state_list,1)

    def lstm_inference(self, extracted_feature, cell_all_hero, hidden_all_hero, if_multi_cell=True):
        if if_multi_cell:
            cell_out_all_hero = []
            hidden_out_all_hero = []
            # feature_with_tsteps=State_splitter().detach_tsteps_dim_from_batch(extracted_feature)
            for i, feat_ph in enumerate(extracted_feature):
                ext_feat_this_hero = self._detach_tsteps(feat_ph)
                cell_in_this_hero = cell_all_hero[i]
                hidden_in_this_hero = hidden_all_hero[i]
                with tf.variable_scope('hero'+f'_{i}_lstm_multi', reuse=self.reuse):
                    cell_out_this_hero, hidden_out_this_hero = self._lstm_multi_cell_forward(
                        self.lstm_cell_ah[i], ext_feat_this_hero, cell_in_this_hero, hidden_in_this_hero)
                cell_out_all_hero.append(cell_out_this_hero)
                hidden_out_all_hero.append(hidden_out_this_hero)
        else:
            cell_out_all_hero = []
            hidden_out_all_hero = []
            for i in range(len(extracted_feature)):
                ext_feat_this_hero = extracted_feature[i]
                cell_in_this_hero = cell_all_hero[i]
                hidden_in_this_hero = hidden_all_hero[i]
                with tf.variable_scope('hero'+f'_{i}_lstm', reuse=self.reuse):
                    cell_out_this_hero, hidden_out_this_hero = self._lstm_cell_forward(self.lstm_cell_ah[i],
                                                                                       ext_feat_this_hero, cell_in_this_hero, hidden_in_this_hero)
                cell_out_all_hero.append(cell_out_this_hero)
                hidden_out_all_hero.append(hidden_out_this_hero)
        return cell_out_all_hero, hidden_out_all_hero

    # LSTM End


class Communication():
    '''Communication between 3 heros'''

    def __init__(self):
        self.score_fc_weight_initializer = Config.score_fc_weight_initializer
        self.reuse = Config.reuse

    def _get_score_shape(self, input_feature_ah):  # ah: all heros; ph: per hero
        dim_list = [feature.get_shape().as_list()[-1]
                    for feature in input_feature_ah]
        return [sum(dim_list), 9]

    def _attention(self, input_feature_ah):
        with tf.variable_scope('Communication', reuse=self.reuse):
            socre_weight_shape = self._get_score_shape(input_feature_ah)
            hero_score = self._fc_weight_variable(
                shape=socre_weight_shape, name="Attention_score_weight")
            hero_feature = tf.concat(
                [input_feature_ah[0], input_feature_ah[1]], axis=1)
            hero_feature = tf.concat(
                [hero_feature, input_feature_ah[2]], axis=1)
            score = tf.nn.tanh(tf.matmul(hero_feature, hero_score))
            score_w = tf.nn.softmax(score, axis=-1)
            feature_data_list0 = input_feature_ah[0] * score_w[:, 0:1] + \
                input_feature_ah[1] * score_w[:, 1:2] + \
                input_feature_ah[2] * score_w[:, 2:3]
            feature_data_list1 = input_feature_ah[0] * score_w[:, 3:4] + \
                input_feature_ah[1] * score_w[:, 4:5] + \
                input_feature_ah[2] * score_w[:, 5:6]
            feature_data_list2 = input_feature_ah[0] * score_w[:, 6:7] + \
                input_feature_ah[1] * score_w[:, 7:8] + \
                input_feature_ah[2] * score_w[:, 8:9]
            return [feature_data_list0, feature_data_list1, feature_data_list2]

    def _fc_weight_variable(self, shape, name, trainable=True):
        #initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.orthogonal_initializer()
        return tf.get_variable(name, shape=shape, initializer=self.score_fc_weight_initializer, trainable=trainable)

    def COMM_inference(self, input_feature_ah):
        return self._attention(input_feature_ah)


class ActionChooser():
    def __init__(self, Config):
        self.Config = Config
        self.reuse = Config.reuse
        self.action_fc_weight_initializer = Config.action_fc_weight_initializer
        self.action_embedding_weight_initializer = Config.action_embedding_weight_initializer
        self.button_num = Config.HERO_LABEL_SIZE_LIST[0][0]
        self.move_num = Config.HERO_LABEL_SIZE_LIST[0][1]
        self.offset_x_num = Config.HERO_LABEL_SIZE_LIST[0][2]
        self.offset_z_num = Config.HERO_LABEL_SIZE_LIST[0][3]
        self.target_num = Config.HERO_LABEL_SIZE_LIST[0][4]

    def _fc_weight_variable(self, shape, name, trainable=True):
        return tf.get_variable(name, shape=shape, initializer=self.action_fc_weight_initializer, trainable=trainable)

    def _embedding_weight_variable(self, shape, name, trainable=True):
        return tf.get_variable(name, shape=shape, initializer=self.action_embedding_weight_initializer, trainable=trainable)
    
    def _fc_bias_variable(self, shape, name, trainable=True):
        return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.0), trainable=trainable)

    def _inference(self, input_feature_ah):
        with tf.variable_scope('ActionChooser', reuse=self.reuse):
            each_hero_action_list = []
            for hero in range(len(input_feature_ah)):
                input_feature_ph = input_feature_ah[hero]
                #import pdb
                #pdb.set_trace()

                # Button choose begin
                # button_fc_shape = [batch_size, EMBEDDING_DIM]
                # button_embedding_weight_shape = [EMBEDDING_DIM, button_num]
                # button_embedding_shape = [batch_size, button_num]
                # button_choice_shape = [batch_size]
                button_fc_weight = self._fc_weight_variable(
                    shape=[input_feature_ph.get_shape().as_list()[-1], self.Config.EMBEDDING_DIM], name=f"hero{hero}_Button_fc_weight")
                button_bias_weight = self._fc_bias_variable(
                    shape=[self.Config.EMBEDDING_DIM], name=f"hero{hero}_Button_bias_weight")
                button_embedding_weight = self._embedding_weight_variable(
                    shape=[self.Config.EMBEDDING_DIM, self.button_num], name=f"hero{hero}_Button_embedding_weight")
                button_fc = tf.matmul(input_feature_ph, button_fc_weight) + button_bias_weight
                button_fc = tf.nn.relu(button_fc)
                button_embedding = tf.matmul(
                    button_fc, button_embedding_weight)
                #button_embedding = tf.nn.softmax(button_embedding, axis=-1)
                button_choice = tf.argmax(button_embedding, axis=-1)
                #button_fc = tf.nn.softmax(button_fc, axis=-1)
                # Button choose end

                # Target choose begin
                # button_choice_embedding_shape = [batch_size, EMBEDDING_DIM]
                # target_embedding_weight_shape = [EMBEDDING_DIM, target_num]
                # button_target_embedding_weight_shape = [batch_size, EMBEDDING_DIM, target_num]
                # target_embedding_shape = [batch_size, self.target_num]
                # target_fc_shape = [batch_size, EMBEDDING_DIM]
                button_choice_embedding = tf.nn.embedding_lookup(
                    tf.transpose(button_embedding_weight, [1, 0]), button_choice)
                target_embedding_weight = self._embedding_weight_variable(
                    shape=[self.Config.EMBEDDING_DIM, self.target_num], name=f"hero{hero}_Target_embedding_weight")
                button_target_embedding_weight = tf.expand_dims(
                    button_choice_embedding, axis=-1) * target_embedding_weight
                target_fc_weight = self._fc_weight_variable(
                    shape=[input_feature_ph.get_shape().as_list()[-1], self.Config.EMBEDDING_DIM], name=f"hero{hero}_Target_fc_weight")
                target_bias_weight = self._fc_bias_variable(
                    shape=[self.Config.EMBEDDING_DIM], name=f"hero{hero}_Target_bias_weight")
                target_fc = tf.matmul(input_feature_ph, target_fc_weight) + target_bias_weight
                target_embedding = tf.matmul(tf.expand_dims(
                    target_fc, axis=1), button_target_embedding_weight)
                target_embedding = tf.squeeze(target_embedding, axis=1)
                #target_embedding = tf.nn.softmax(target_embedding, axis=-1)
                target_choice = tf.argmax(target_embedding, axis=-1)
                #target_fc = tf.nn.softmax(target_fc, axis=-1)
                # Target choose end

                # Move choose begin
                # move_choice_shape = [batch_size]
                move_fc_weight = self._fc_weight_variable(
                    shape=[input_feature_ph.get_shape().as_list()[-1], self.move_num], name=f"hero{hero}_Move_fc_weight")
                move_bias_weight = self._fc_bias_variable(
                    shape=[self.move_num], name=f"hero{hero}_Move_bias_weight")
                move_fc = tf.matmul(input_feature_ph, move_fc_weight) + move_bias_weight
                #move_fc = tf.nn.softmax(move_fc, axis=-1)
                move_choice = tf.argmax(move_fc, axis=-1)
                # Move choose end

                # Offset choose begin
                # offset_x_choice_shape = [batch_size]
                # offset_z_choice_shape = [batch_size]
                offset_x_fc_weight = self._fc_weight_variable(
                    shape=[input_feature_ph.get_shape().as_list()[-1], self.offset_x_num], name=f"hero{hero}_Offset_x_fc_weight")
                offset_x_bias_weight = self._fc_bias_variable(
                    shape=[self.offset_x_num], name=f"hero{hero}_Offset_x_bias_weight")
                offset_x_fc = tf.matmul(
                    input_feature_ph, offset_x_fc_weight) + offset_x_bias_weight
                #offset_x_fc = tf.nn.softmax(offset_x_fc, axis=-1)
                offset_x_choice = tf.argmax(offset_x_fc, axis=-1)
                offset_z_fc_weight = self._fc_weight_variable(
                    shape=[input_feature_ph.get_shape().as_list()[-1], self.offset_z_num], name=f"hero{hero}_Offset_z_fc_weight")
                offset_z_bias_weight = self._fc_bias_variable(
                    shape=[self.offset_z_num], name=f"hero{hero}_Offset_z_bias_weight")
                offset_z_fc = tf.matmul(
                    input_feature_ph, offset_z_fc_weight) + offset_z_bias_weight
                #offset_z_fc = tf.nn.softmax(offset_z_fc, axis=-1)
                offset_z_choice = tf.argmax(offset_z_fc, axis=-1)
                # Offset choose end

                # Network value
                network_value_fc_weight = self._fc_weight_variable(
                    shape=[input_feature_ph.get_shape().as_list()[-1], 1], name=f"hero{hero}_Network_value_fc_weight")
                network_value_bias_weight = self._fc_bias_variable(
                    shape=[1], name=f"hero{hero}_Network_value_bias_weight")
                network_value = tf.matmul(
                    input_feature_ph, network_value_fc_weight) + network_value_bias_weight

                each_hero_action_list.append([
                    button_embedding,
                    move_fc,
                    offset_x_fc,
                    offset_z_fc,
                    target_embedding,
                    network_value])
            return each_hero_action_list

    def Action_inference(self, input_feature_ah):
        return self._inference(input_feature_ah)
        