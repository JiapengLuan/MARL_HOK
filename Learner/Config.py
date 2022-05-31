class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16 # lstm step
    LSTM_UNIT_SIZE = 128*3 # lstm cell and hidden state size
    HERO_DATA_SPLIT_SHAPE = [
                             [4586,  13, 25, 42, 42, 39,  1,  1,  1, 1, 1, 1, 1,  13, 25, 42, 42, 39,  1,  1,1,1,1,1],
                             [4586,  13, 25, 42, 42, 39,  1,  1,  1, 1, 1, 1, 1,  13, 25, 42, 42, 39,  1,  1,1,1,1,1],
                             [4586,  13, 25, 42, 42, 39,  1,  1,  1, 1, 1, 1, 1,  13, 25, 42, 42, 39,  1,  1,1,1,1,1]
                            ] # idx0:feature, idx1-5:legal_action, idx6:reward_sum, idx7:advantage, idx8-12:action_list, idx13-17:prob_list, idx18:is_train, idx19-24: train_list
    HERO_SERI_VEC_SPLIT_SHAPE = [
                                 [(6, 17, 17), (2852,)],
                                 [(6, 17, 17), (2852,)],
                                 [(6, 17, 17), (2852,)]
                                ] # image data shape and vec data shape for each hero
    HERO_FEATURE_IMG_CHANNEL = [[6], [6], [6]] # feature image channel for each hero
    HERO_LABEL_SIZE_LIST = [
                             [13, 25, 42, 42, 39],
                             [13, 25, 42, 42, 39],
                             [13, 25, 42, 42, 39]
                           ] # action label list for each hero
    HERO_IS_REINFORCE_TASK_LIST = [
                                   [True, True, True, True, True],
                                   [True, True, True, True, True],
                                   [True, True, True, True, True]
                                  ] # if action label need to participate in training

    INIT_LEARNING_RATE_START = 0.0006
    BETA_START = 0.008
    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.01
    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001
    TASK_ID = 16980
    TASK_UUID = "10ad0318-893f-4426-ac8e-44f109561350"
    data_keys = "hero1_data,hero2_data,hero3_data"
    data_shapes = [[sum(HERO_DATA_SPLIT_SHAPE[0])*LSTM_TIME_STEPS + LSTM_UNIT_SIZE*2]] * 3
    import tensorflow as tf
    BN_EPSILON=0.001
    resnet_FeatureImgLikeMg_n=1 #num of residual blocks used for FeatureImgLike per hero
    reuse = tf.compat.v1.AUTO_REUSE #REUSE parameter for tf.variable_scope()
    states_names=['FeatureImgLikeMg','VecFeatureHero','MainHeroFeature','VecSoldier','VecOrgan','VecMonster','VecCampsWholeInfo']


    img_num_res_blocks=1
    img_net_type='img_conv'#'img_conv' or 'img_res'
    vec_feat_extract_out_dims=[[8,8,8,8,8,8],  #[[1st fc],[2nd fc]]
                                [8,8,8,8,8,8]]
    vec_fc_2ndlayer_type='resfc'#'resfc' or 'fc' 
    if_vec_fc_bn=True #Should vec 2 fc layers contain bn layer? When 2nd layer is 'resfc', this had better be True; when 2nd layer is 'fc', we can try True or False
    num_vec_fc_in_resblock=1 
    vecNet_fc_initializer=tf.orthogonal_initializer()
    
    EMBEDDING_DIM = 16

    score_fc_weight_initializer=tf.orthogonal_initializer()
    action_fc_weight_initializer = tf.orthogonal_initializer()
    action_embedding_weight_initializer = tf.orthogonal_initializer()