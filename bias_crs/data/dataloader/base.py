# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import random
from abc import ABC
from copy import copy
from loguru import logger
from math import ceil
from tqdm import tqdm
from bias_crs.config import DATA_PATH
from bias_crs.config import ROOT_PATH, DATASET_PATH
import json
import os
from datetime import datetime
from copy import deepcopy
from operator import add
import numpy as np

class BaseDataLoader(ABC):
    """Abstract class of dataloader

    Notes:
        ``'scale'`` can be set in config to limit the size of dataset.

    """

    def __init__(self, opt, dataset, aug_dataset=None, item_popularity=None):
        """
        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: dataset

        """
        self.opt = opt
        self.dataset = dataset
        self.scale = opt.get('scale', 1)
        assert 0 < self.scale <= 1
    
    def get_data(self, batch_fn, batch_size, shuffle=True, process_fn=None):
        """Collate batch data for system to fit

        Args:
            batch_fn (func): function to collate data
            batch_size (int):
            shuffle (bool, optional): Defaults to True.
            process_fn (func, optional): function to process dataset before batchify. Defaults to None.

        Yields:
            tuple or dict of torch.Tensor: batch data for system to fit

        """
        dataset = self.dataset
        if process_fn is not None:
            dataset = process_fn()
            logger.info('[Finish dataset process before batchify]')
        dataset = dataset[:ceil(len(dataset) * self.scale)]
        logger.debug(f'[Dataset size: {len(dataset)}]')

        batch_num = ceil(len(dataset) / batch_size)
        idx_list = list(range(len(dataset)))
        if shuffle:
            random.shuffle(idx_list)

        for start_idx in tqdm(range(batch_num)):
            batch_idx = idx_list[start_idx * batch_size: (start_idx + 1) * batch_size]
            batch = [dataset[idx] for idx in batch_idx]
            batch = batch_fn(batch)
            if batch == False:
                continue
            else:
                yield(batch) 
    
    
    
    def _raw_aug_data_process(self, raw_data):
        if self.dataset_name == 'tgredial':
            self.pad_topic_idx = 0
            self.unk_token_idx = 3
            self.topic2ind = json.load(open(os.path.join(self.dpath, 'topic2id.json'), 'r', encoding='utf-8'))
            self.replace_token = self.opt.get('replace_token',None)
            self.replace_token_idx = self.opt.get('replace_token_idx',None)
            self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
            self.entity2id = json.load(open(os.path.join(self.dpath, 'entity2id.json'), encoding='utf-8'))  # {entity: entity_id}
            self.word2id = json.load(open(os.path.join(self.dpath, 'word2id.json'), 'r', encoding='utf-8'))
            self.conv2history = json.load(open(os.path.join(self.dpath, 'user2history.json'), 'r', encoding='utf-8'))
            self.user2profile = json.load(open(os.path.join(self.dpath, 'user2profile.json'), 'r', encoding='utf-8'))
            augmented_convs = {item_id: self._convert_to_id(conversation) for item_id, conversation in raw_data.items()}
            augmented_conv_dicts = {item_id: self._augment_and_add(conv) for item_id, conv in augmented_convs.items()}
            # if self.model == 'UCCR':
            #     augmented_conv_dicts = self.process_uccr_data(augmented_convs)
        elif self.dataset_name == 'redial':
            self.unk_token_idx = 3
            self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
            self.entity2id = json.load(
                open(os.path.join(self.dpath, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
            self.word2id = json.load(open(os.path.join(self.dpath, 'concept2id.json'), 'r', encoding='utf-8'))

            augmented_convs = {item_id: self._merge_conv_data(conversation) for item_id, conversation in raw_data.items()}
            augmented_conv_dicts = {item_id: self._augment_and_add(conversation) for item_id, conversation in augmented_convs.items()}
        return augmented_conv_dicts
    
    
    def _convert_to_id(self, conversation):
        augmented_convs = []
        last_role = None
        # print('conversation keys: ', conversation.keys()) # conv_id', 'messages', 'user_id' 
        for utt in conversation['messages']:
            # if utt['role'] == last_role:
            #     print(utt)
            # assert utt['role'] != last_role
            # change movies into slots
            if self.replace_token:
                if len(utt['movie']) != 0:
                    while  '《' in utt['text'] :
                        begin = utt['text'].index("《")
                        end = utt['text'].index("》")
                        utt['text'] = utt['text'][:begin] + [self.replace_token] + utt['text'][end+1:]
            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            movie_ids = [self.entity2id[movie] for movie in utt['movie'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]
            policy = []
            for action, kw in zip(utt['target'][1::2], utt['target'][2::2]):
                if kw is None or action == '推荐电影':
                    continue
                if isinstance(kw, str):
                    kw = [kw]
                kw = [self.topic2ind.get(k, self.pad_topic_idx) for k in kw]
                policy.append([action, kw])
            final_kws = [self.topic2ind[kw] if kw is not None else self.pad_topic_idx for kw in utt['final'][1]]
            final = [utt['final'][0], final_kws]
            conv_utt_id = str(conversation['conv_id']) + '/' + str(utt['local_id'])
            interaction_history = self.conv2history.get(conv_utt_id, [])
            user_profile = self.user2profile[conversation['user_id']] if conversation['user_id'] != 'syn' else []
            user_profile = [[self.tok2ind.get(token, self.unk_token_idx) for token in sent] for sent in user_profile]

            augmented_convs.append({
                "conv_id": conversation['conv_id'],
                "user_id": conversation['user_id'],
                "role": utt["role"],
                "text": text_token_ids,
                "entity": entity_ids,
                "movie": movie_ids,
                "word": word_ids,
                'policy': policy,
                'final': final,
                'interaction_history': interaction_history,
                'user_profile': user_profile
            })
            last_role = utt["role"]

        return augmented_convs
    
    
    def _merge_conv_data(self, conversation):
        augmented_convs = []
        last_role = None
        dialog = conversation['dialog']
        for utt in dialog:
            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            movie_ids = [self.entity2id[movie] for movie in utt['movies'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += text_token_ids
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
                augmented_convs[-1]["word"] += word_ids
            else:
                augmented_convs.append({
                    "conv_id": conversation['conv_id'],
                    "role": utt["role"],
                    "text": text_token_ids,
                    "entity": entity_ids,
                    "movie": movie_ids,
                    "word": word_ids
                })
            last_role = utt["role"]

        return augmented_convs
    
    def _augment_and_add(self, raw_conv_dict):
        if self.dataset_name == 'redial':
            # print("it goes here redial")
            augmented_conv_dicts = []
            context_tokens, context_entities, context_words, context_items = [], [], [], []
            entity_set, word_set = set(), set()
            for i, conv in enumerate(raw_conv_dict):
                text_tokens, entities, movies, words = conv["text"], conv["entity"], conv["movie"], conv["word"]
                if len(context_tokens) > 0:
                    conv_dict = {
                        "role": conv['role'],
                        "conv_id": conv['conv_id'],
                        "user_id": 'syn_user',
                        "context_tokens": copy(context_tokens),
                        "response": text_tokens,
                        "context_entities": copy(context_entities),
                        "context_words": copy(context_words),
                        "context_items": copy(context_items),
                        "items": movies,
                    }
                    augmented_conv_dicts.append(conv_dict)

                context_tokens.append(text_tokens)
                context_items += movies
                for entity in entities + movies:
                    if entity not in entity_set:
                        entity_set.add(entity)
                        context_entities.append(entity)
                for word in words:
                    if word not in word_set:
                        word_set.add(word)
                        context_words.append(word)
        elif self.dataset_name == 'tgredial':
            # print("it goes here tgredial")
            augmented_conv_dicts = []
            context_tokens, context_entities, context_words, context_policy, context_items = [], [], [], [], []
            context_entities_pos, context_words_pos = [], []
            entity_set, word_set = set(), set()
            for i, conv in enumerate(raw_conv_dict):
                text_tokens, entities, movies, words, policies = conv["text"], conv["entity"], conv["movie"], conv["word"], \
                                                                 conv['policy']
                if self.replace_token is not None: 
                    if text_tokens.count(30000) != len(movies):
                        continue # the number of slots doesn't equal to the number of movies

                if len(context_tokens) > 0:
                    if self.model == 'UCCR':
                        current_conv_interaction = dict()
                        current_conv_interaction['history_entity']= [context_entities]
                        current_conv_interaction['history_entities_pos']= [context_entities_pos]
                        current_conv_interaction['history_word']= [context_words]
                        current_conv_interaction['history_words_pos']= [context_words_pos]
                        current_conv_interaction['history_item'] = [context_items]
                        conv_dict = {
                        'role': conv['role'],
                        'time_id': 0,
                        "conv_id": conv['conv_id'],
                        "user_id": conv['user_id'],
                        'user_profile': conv['user_profile'],
                        "context_tokens": copy(context_tokens),
                        "response": text_tokens,
                        "context_entities": copy(context_entities),
                        "context_entities_pos": copy(context_entities_pos),
                        "context_words": copy(context_words),
                        "context_words_pos": copy(context_words_pos),
                        'interaction_history': conv['interaction_history'],
                        'context_items': copy(context_items),
                        "items": movies,
                        "histories": [current_conv_interaction],
                        'context_policy': copy(context_policy),
                        'target': policies,
                        'final': conv['final'],

                    }
                    else:
                        conv_dict = {
                            'role': conv['role'],
                            "conv_id": conv['conv_id'],
                            "user_id": conv['user_id'],
                            'user_profile': conv['user_profile'],
                            "context_tokens": copy(context_tokens),
                            "response": text_tokens,
                            "context_entities": copy(context_entities),
                            "context_entities_pos": copy(context_entities_pos),
                            "context_words": copy(context_words),
                            "context_words_pos": copy(context_words_pos),
                            'interaction_history': conv['interaction_history'],
                            'context_items': copy(context_items),
                            "items": movies,
                            'context_policy': copy(context_policy),
                            'target': policies,
                            'final': conv['final'],

                        }
                    augmented_conv_dicts.append(conv_dict)

                context_tokens.append(text_tokens)
                context_policy.append(policies)
                context_items += movies
                for entity in entities + movies:
                    if entity not in entity_set:
                        entity_set.add(entity)
                        context_entities.append(entity)
                        context_entities_pos.append(i)
                for word in words:
                    if word not in word_set:
                        word_set.add(word)
                        context_words.append(word)
                        context_words_pos.append(i)
        return augmented_conv_dicts
    
    def load_aug_data(self):
        
        tokenize = self.opt['tokenize']['rec']
        # tokenize = self.opt['tokenize']
        # print("DATASET_PATH", DATASET_PATH)
        self.dataset_name = 'redial'
        self.model = self.opt['model']
        self.dpath = os.path.join(DATASET_PATH, self.dataset_name, tokenize)
        with open(os.path.join(self.dpath, 'aug_data.json'), 'r', encoding='utf-8') as f:
            aug_dataset = json.load(f)
            logger.debug(f"[Load augmented data from {os.path.join(self.dpath, 'aug_data.json')}]")
        self.aug_dataset_processed = self._raw_aug_data_process(aug_dataset)
        with open(os.path.join(self.dpath, 'movie_frequency.json'), 'r', encoding='utf-8') as f:
            self.item_popularity = json.load(f)
            self.item_popularity = {k: v for k, v in sorted(self.item_popularity.items(), key=lambda item: item[1], reverse=True)}
            self.ranked_items = list(self.item_popularity.keys())
            self.ranked_items_index = {k: v for v, k in enumerate(self.ranked_items)}
            self.item_weights = list(self.item_popularity.values())
            logger.debug(f"[Load item popularity data from {os.path.join(self.dpath, 'movie_frequency.json')}]")
        with open(os.path.join(self.dpath, 'movie_ids.json'), 'r', encoding='utf-8') as f: 
            self.movie_entity_ids = json.load(f)
            logger.debug(f"[Load movie entity ids from {os.path.join(self.dpath, 'movie_ids.json')}]")
        
    
    def get_aug_data(self, item_id):
        # get popularity ranked item list
        # item popularity  
        # consider lower popularity items
        # print("item_id: ", item_id)
        # ranked_items = list(self.item_popularity.keys())
        # item_weights = list(self.item_popularity.values())
        # item_index = self.ranked_items.index(str(item_id))
        file = open(os.path.join(self.dpath, 'redial_0_v2.txt'), 'a')
        file.write(str(item_id) + '\n')
        item_index = self.ranked_items_index[str(item_id)]
        sampled_item = random.choices(self.ranked_items[(item_index + 1):], 
                                      self.item_weights[(item_index + 1):], k=1)
        # for item in sampled_item:
        #     file.write(str(item) + '\n')
        file.close()
        aug_data = [self.aug_dataset_processed[str(item)] for item in sampled_item if str(item) in self.aug_dataset_processed]  
        return aug_data
    
    
    def process_aug(self, dataset):
        if self.dataset_name == 'redial':
            augment_dataset = []
            for conversation in dataset:
                if conversation['role'] == 'Recommender':
                    for item in conversation['items']:
                        context_entities = conversation['context_entities']
                        augment_dataset.append({'context_entities': context_entities, 
                                        'item': item,
                                        'role': conversation['role'],
                                        'conv_id': conversation['conv_id'],
                                        'user_id': conversation['user_id'],
                                        'context_tokens': conversation['context_tokens'],
                                        'response': conversation['response'],
                                        'context_words': conversation['context_words'],
                                        'context_items': conversation['context_items']
                                       })
        elif self.dataset_name == 'tgredial':
            augment_dataset = []
            for conv_dict in dataset:
                for movie in conv_dict['items']:
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['item'] = movie
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset
    
    
    def neg_idx(self, start_idx, batch_num, neg_samples=5):
        try:
            neg_idx = np.random.randint(0, batch_num-2, size=neg_samples)
        except:
            neg_idx = np.random.randint(0,1,size=5)
        if start_idx in neg_idx:
            neg_idx = [(x+1)%(batch_num-2) for x in neg_idx]
        return neg_idx
        
    
    def get_rec_aug_data(self, batch_fn, batch_size, shuffle=True, process_fn=None):
        """Collate batch data for system to fit

        Args:
            batch_fn (func): function to collate data
            batch_size (int):
            shuffle (bool, optional): Defaults to True.
            process_fn (func, optional): function to process dataset before batchify. Defaults to None.

        Yields:
            tuple or dict of torch.Tensor: batch data for system to fit

        """
        dataset = self.dataset
        if process_fn is not None:
            dataset = process_fn()
            logger.info('[Finish dataset process before batchify]')
        dataset = dataset[:ceil(len(dataset) * self.scale)]
        logger.debug(f'[Dataset size: {len(dataset)}]')

        batch_num = ceil(len(dataset) / batch_size)
        idx_list = list(range(len(dataset)))
        if shuffle:
            random.shuffle(idx_list)
        # print("batch_num: ", batch_num)
        for start_idx in tqdm(range(batch_num)):
            batch_idx = idx_list[start_idx * batch_size: (start_idx + 1) * batch_size]
            batch = [dataset[idx] for idx in batch_idx]
            items = [conversation['item'] for conversation in batch] # item is the item_index
            
            # version 1
            # print("get aug_data: ", datetime.now())
            # item_indexes = [self.ranked_items_index[str(item)] for item in items]
            # # randomly move to less popular items in a range of 1 to 100.
            # move_step = np.random.randint(1,100, size=len(item_indexes))
            # aug_data_index = list(map(add, item_indexes, move_step))
            # aug_items = [self.ranked_items[idx] for idx in aug_data_index]
            # aug_dialog = [self.process_aug(self.aug_dataset_processed[str(item)]) for item in aug_items if str(item) in self.aug_dataset_processed]
            # # print("finish: ", datetime.now())
            # # print("batch size: ", len(batch))
            # for dialog in aug_dialog:
            #     if len(dialog) != 0:
            #         batch.extend(dialog)
            
            # version 2
            # print("get aug_data: ", datetime.now())
            # print("number of items to be augmented: ", len(items))
            
            for item in items:
                aug_dialog = self.get_aug_data(item)
                for dialog in aug_dialog:
                    aug_data = self.process_aug(dialog)
                    if len(aug_data) == 0:
                        continue
                    else:
                        batch.extend(aug_data)
            
            if self.model == 'UCCR':
                batch_size = len(batch)
                neg_idx_list = self.neg_idx(start_idx, batch_num)
                neg_batch_idx = [idx_list[x * batch_size: (x + 1) * batch_size] for x in neg_idx_list]
                neg_batches = [[dataset[y] for y in x] for x in neg_batch_idx]
                yield batch_fn(batch, neg_batches)
            else:
                batch = batch_fn(batch) # finished batchify
                if batch == False:
                    continue
                else:
                    # print("init_batch: ", batch)
                    yield(batch) 

    def get_conv_data(self, batch_size, shuffle=True):
        """get_data wrapper for conversation.

        You can implement your own process_fn in ``conv_process_fn``, batch_fn in ``conv_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for conversation.

        """
        return self.get_data(self.conv_batchify, batch_size, shuffle, self.conv_process_fn)

    def get_rec_data(self, batch_size, batch_mode='basic', shuffle=True):
        """get_data wrapper for recommendation.

        You can implement your own process_fn in ``rec_process_fn``, batch_fn in ``rec_batchify``.

        Args:
            batch_size (int):
            batch_mode (str): ['basic', 'popnudge']
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for recommendation.

        """
        assert batch_mode in ['basic', 'popnudge']
        if batch_mode is 'basic':
            return self.get_data(self.rec_batchify, batch_size, shuffle, self.rec_process_fn)
        elif batch_mode is 'popnudge':
            self.load_aug_data()
            return self.get_rec_aug_data(self.rec_batchify, batch_size, shuffle, self.rec_process_fn)

    def get_policy_data(self, batch_size, shuffle=True):
        """get_data wrapper for policy.

        You can implement your own process_fn in ``self.policy_process_fn``, batch_fn in ``policy_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for policy.

        """
        return self.get_data(self.policy_batchify, batch_size, shuffle, self.policy_process_fn)

    def conv_process_fn(self):
        """Process whole data for conversation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def conv_batchify(self, batch):
        """batchify data for conversation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train conversation part.
        """
        raise NotImplementedError('dataloader must implement conv_batchify() method')

    def rec_process_fn(self):
        """Process whole data for recommendation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def rec_batchify(self, batch):
        """batchify data for recommendation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train recommendation part.
        """
        raise NotImplementedError('dataloader must implement rec_batchify() method')

    def policy_process_fn(self):
        """Process whole data for policy before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def policy_batchify(self, batch):
        """batchify data for policy after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train policy part.
        """
        raise NotImplementedError('dataloader must implement policy_batchify() method')

    def retain_recommender_target(self):
        """keep data whose role is recommender.

        Returns:
            Recommender part of ``self.dataset``.

        """
        dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                dataset.append(conv_dict)
        return dataset

    def rec_interact(self, data):
        """process user input data for system to recommend.

        Args:
            data: user input data.

        Returns:
            data for system to recommend.
        """
        pass

    def conv_interact(self, data):
        """Process user input data for system to converse.

        Args:
            data: user input data.

        Returns:
            data for system in converse.
        """
        pass
