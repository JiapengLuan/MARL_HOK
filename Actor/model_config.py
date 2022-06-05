# -*- coding:utf-8 -*-

class ModelConfig:
    HERO_NUM = 3
    LSTM_TIME_STEPS = 1
    LSTM_UNIT_SIZE = 128*3
    HERO_DATA_SPLIT_SHAPE = [
                             [4586,  13, 25, 42, 42, 39,  1,  1,  1, 1, 1, 1, 1,  13, 25, 42, 42, 39,  1,  1,1,1,1,1],
                             [4586,  13, 25, 42, 42, 39,  1,  1,  1, 1, 1, 1, 1,  13, 25, 42, 42, 39,  1,  1,1,1,1,1],
                             [4586,  13, 25, 42, 42, 39,  1,  1,  1, 1, 1, 1, 1,  13, 25, 42, 42, 39,  1,  1,1,1,1,1]
                            ]

    HERO_SERI_VEC_SPLIT_SHAPE = [
                                 [(6, 17, 17), (2852,)],
                                 [(6, 17, 17), (2852,)],
                                 [(6, 17, 17), (2852,)]
                                ]
    HERO_LABEL_SIZE_LIST = [
                             [13, 25, 42, 42, 39],
                             [13, 25, 42, 42, 39],
                             [13, 25, 42, 42, 39]
                           ]
    HERO_IS_REINFORCE_TASK_LIST = [
                                   [True, True, True, True, True],
                                   [True, True, True, True, True],
                                   [True, True, True, True, True]
                                  ]

    INIT_LEARNING_RATE_START = 0.0006
    BETA_START = 0.008
    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001
    TASK_ID = 16980
    TASK_UUID = "10ad0318-893f-4426-ac8e-44f109561350"
    data_shapes = [[sum(HERO_DATA_SPLIT_SHAPE[0])*LSTM_TIME_STEPS + LSTM_UNIT_SIZE*2]] * 3
    
    import tensorflow as tf
    BN_EPSILON=0.001
    resnet_FeatureImgLikeMg_n=1 #num of residual blocks used for FeatureImgLike per hero
    reuse = tf.compat.v1.AUTO_REUSE #REUSE parameter for tf.variable_scope()
    states_names=['FeatureImgLikeMg','VecFeatureHero','MainHeroFeature','VecSoldier','VecOrgan','VecMonster','VecCampsWholeInfo']


    img_num_res_blocks=1
    img_net_type='img_conv'#'img_conv' or 'img_res'
    vec_feat_extract_out_dims=[[64,64,64,64,64,None],  #[[1st fc],[2nd fc],[3nd fc]]
                                [32,32,32,32,32,None],
                                [32,32,32,32,32,32]]
    vec_unit_group_split_id=[3,None,10,3,10,None] #indeces for splitting units into two sides
    vec_fc_2ndlayer_type='fc'#'resfc' or 'fc' 
    if_vec_fc_bn=False #Should vec 2 fc layers contain bn layer? When 2nd layer is 'resfc', this had better be True; when 2nd layer is 'fc', we can try True or False
    num_vec_fc_in_resblock=1 
    vecNet_fc_weight_initializer=tf.orthogonal_initializer()
    vecNet_fc_bias_initializer=tf.constant_initializer(0.0)
    img_conv_initializer=tf.orthogonal_initializer()
    
    EMBEDDING_DIM = 16

    score_fc_weight_initializer=tf.orthogonal_initializer()
    action_fc_weight_initializer = tf.orthogonal_initializer()
    action_embedding_weight_initializer = tf.orthogonal_initializer()