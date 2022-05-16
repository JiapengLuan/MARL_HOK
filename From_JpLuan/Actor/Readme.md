main code for generating extracted features for 3 heros is
        whole_feature_list = State_splitter().split_features(each_hero_data_list)
        extracted_feature = Feature_extraction().get_extracted_feature(whole_feature_list),
which is in
def _inference(self, each_hero_data_list, only_inference=True):
