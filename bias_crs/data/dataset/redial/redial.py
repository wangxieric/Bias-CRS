# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2021/1/3, 2020/12/19
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail

r"""
ReDial
======
References:
    Li, Raymond, et al. `"Towards deep conversational recommendations."`_ in NeurIPS 2018.

.. _`"Towards deep conversational recommendations."`:
   https://papers.nips.cc/paper/2018/hash/800de15c79c8d840f4e78d3af937d4d4-Abstract.html

"""

import json
import os
from collections import defaultdict
from copy import copy
import numpy as np
from loguru import logger
from tqdm import tqdm

from bias_crs.config import ROOT_PATH, DATASET_PATH
from bias_crs.data.dataset.base import BaseDataset
from .resources import resources

import pickle as pkl


class ReDialDataset(BaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.
        vocab (dict): ::

            {
                'tok2ind': map from token to index,
                'ind2tok': map from index to token,
                'entity2id': map from entity to index,
                'id2entity': map from index to entity,
                'word2id': map from word to index,
                'vocab_size': len(self.tok2ind),
                'n_entity': max(self.entity2id.values()) + 1,
                'n_word': max(self.word2id.values()) + 1,
            }

    Notes:
        ``'unk'`` must be specified in ``'special_token_idx'`` in ``resources.py``.

    """

    def __init__(self, opt, tokenize, restore=False, save=False):
        """Specify tokenized resource and init base dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            tokenize (str): how to tokenize dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.
            model (str): enable a specific loading of data for a CRS model.
        """
        resource = resources[tokenize]
        self.special_token_idx = resource['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']
        dpath = os.path.join(DATASET_PATH, "redial", tokenize)
        self.opt = opt
        # load specific data for models
        if self.opt["rec_model"] == 'RevCoreRec':
            self.subkg = pkl.load(open(os.path.join(ROOT_PATH, "bias_crs/data/dataset/redial", 'revcore_data/subkg.pkl'), 'rb'))
            self.text_dict = pkl.load(open(os.path.join(ROOT_PATH, "bias_crs/data/dataset/redial", 'revcore_data/text_dict_new.pkl'), 'rb'))
            self.reviews  = pkl.load(open(os.path.join(ROOT_PATH, "bias_crs/data/dataset/redial", 'revcore_data/movie2tokenreview_helpful.pkl'), 'rb'))
            # prepare word2vec
            self.word2index = json.load(open(os.path.join(ROOT_PATH, "bias_crs/data/dataset/redial", 'revcore_data/word2index_redial2.json'), 
                            encoding='utf-8'))
            self.key2index = json.load(open(os.path.join(ROOT_PATH, "bias_crs/data/dataset/redial", 'revcore_data/key2index_3rd.json'),
                            encoding='utf-8'))
            self.stopwords = set([word.strip() for word in open(os.path.join(ROOT_PATH, "bias_crs/data/dataset/redial",'revcore_data/stopwords.txt'),
                            encoding='utf-8')])
            self.full_data = self._load_full_data(os.path.join(ROOT_PATH, "bias_crs/data/dataset/redial", 'full_raw_data/'))
            self.corpus = []
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
         

        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'word2id': self.word2id,
            'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity,
            'n_word': self.n_word,
        }

        vocab.update(self.special_token_idx)

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.dpath, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        self.entity_max=len(self.entity2id)
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = json.load(open(os.path.join(self.dpath, 'dbpedia_subkg.json'), 'r', encoding='utf-8'))
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'dbpedia_subkg.json')}]")

        # conceptNet
        # {concept: concept_id}
        self.word2id = json.load(open(os.path.join(self.dpath, 'concept2id.json'), 'r', encoding='utf-8'))
        self.n_word = max(self.word2id.values()) + 1
        # {relation\t concept \t concept}
        self.word_kg = open(os.path.join(self.dpath, 'conceptnet_subkg.txt'), 'r', encoding='utf-8')
        logger.debug(
            f"[Load word dictionary and KG from {os.path.join(self.dpath, 'concept2id.json')} and {os.path.join(self.dpath, 'conceptnet_subkg.txt')}]")

    def _load_full_data(self, data_dir):
        # combine all the data to identify missing data from initial datasets
        data = []
        f = open(data_dir + 'train_data.jsonl')
        for line in f:
            data.append(json.loads(line))
        f = open(data_dir + 'test_data.jsonl')
        for line in f:
            data.append(json.loads(line))
        conv_dict = {}
        for conv in data:
            conv_dict[conv['conversationId']] = conv
        return conv_dict

    def _data_preprocess(self, train_data, valid_data, test_data):
        processed_train_data = self._raw_data_process(train_data)
        logger.debug("[Finish train data process]")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.debug("[Finish valid data process]")
        processed_test_data = self._raw_data_process(test_data)
        logger.debug("[Finish test data process]")
        processed_side_data = self._side_data_process()
        logger.debug("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
        augmented_convs = {conversation['conv_id']: self._merge_conv_data(conversation["dialog"]) for conversation in tqdm(raw_data)}
        augmented_conv_dicts = []
        for conv_id, conv in tqdm(augmented_convs.items()):
            augmented_conv_dicts.extend(self._augment_and_add(conv_id, conv))
        return augmented_conv_dicts

    def _merge_conv_data(self, dialog):
        augmented_convs = []
        last_role = None
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
                    "role": utt["role"],
                    "text": text_token_ids,
                    "entity": entity_ids,
                    "movie": movie_ids,
                    "word": word_ids
                })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, conv_id, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_tokens_text, context_entities, context_words, context_items = [], [], [], [], []
        entity_set, word_set = set(), set()
        contexts = self.full_data[conv_id]['messages']
        for idx, conv in enumerate(raw_conv_dict):
            # text is the split full words and the words are the ones that removed some stop words
            text_tokens, entities, movies, words = conv["text"], conv["entity"], conv["movie"], conv["word"]
            text_tokens_text = [self.ind2tok[text_token] for text_token in text_tokens]
            if len(context_tokens) > 0:
                context,c_lengths,concept_mask, dbpedia_mask, reviews_mask,_=self.padding_context(context_tokens_text)
                concept_vec=np.zeros(self.opt['n_concept'] + 1)
                for con in concept_mask:
                    if con!=0:
                        concept_vec[con]=1
                db_vec=np.zeros(self.opt['n_entity'])
                for db in dbpedia_mask:
                    if db!=0:
                        db_vec[db]=1
                entity_vec = np.zeros(self.opt['n_entity'])
                entity_vector=np.zeros(50, dtype=np.int)
                point = 0
                for en in entities:
                    entity_vec[en]=1
                    entity_vector[point]=en
                    point += 1
                response,_,_,_,_ = self.padding_w2v(text_tokens_text, self.opt['max_r_length'])
                conv_dict = {
                    "role": conv['role'],
                    "user": contexts[idx]['senderWorkerId'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    "context_items": copy(context_items),
                    "items": movies,
                    "mask_response": response,
                    "concept_mask": concept_mask,
                    "concept_vec": concept_vec,
                    "reviews_mask": reviews_mask,
                    "db_vec": db_vec,
                    "entity_vector": entity_vector,
                }
                augmented_conv_dicts.append(conv_dict)
            context_tokens.append(text_tokens)
            context_tokens_text.append(text_tokens_text)
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
            for word in words:
                if word not in word_set:
                    word_set.add(word)
                    context_words.append(word)
            if self.opt["rec_model"] == 'RevCoreRec':
                self.corpus.append(' '.join(text_tokens_text))
        return augmented_conv_dicts

    def _side_data_process(self):
        processed_entity_kg = self._entity_kg_process()
        logger.debug("[Finish entity KG process]")
        processed_word_kg = self._word_kg_process()
        logger.debug("[Finish word KG process]")
        movie_entity_ids = json.load(open(os.path.join(self.dpath, 'movie_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load movie entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "item_entity_ids": movie_entity_ids,
        }
        return side_data

    def _entity_kg_process(self, SELF_LOOP_ID=185):
        edge_list = []  # [(entity, entity, relation)]
        for entity in range(self.n_entity):
            if str(entity) not in self.entity_kg:
                continue
            edge_list.append((entity, entity, SELF_LOOP_ID))  # add self loop
            for tail_and_relation in self.entity_kg[str(entity)]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                edges.add((h, t, relation2id[r]))
                entities.add(self.id2entity[h])
                entities.add(self.id2entity[t])
        return {
            'edge': list(edges),
            'n_relation': len(relation2id),
            'entity': list(entities)
        }

    def _word_kg_process(self):
        edges = set()  # {(entity, entity)}
        entities = set()
        for line in self.word_kg:
            kg = line.strip().split('\t')
            entities.add(kg[1].split('/')[0])
            entities.add(kg[2].split('/')[0])
            e0 = self.word2id[kg[1].split('/')[0]]
            e1 = self.word2id[kg[2].split('/')[0]]
            edges.add((e0, e1))
            edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return {
            'edge': list(edges),
            'entity': list(entities)
        }

    def padding_context(self, contexts, pad=0, transformer=True):
        vectors=[]
        vec_lengths=[]
        if transformer==False:
            if len(contexts) > self.max_count:
                for sen in contexts[-self.opt['max_count']:]:
                    vec, v_l = self.padding_w2v(sen, self.opt['max_r_length'], transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors,vec_lengths,self.max_count
            else:
                length=len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen, self.opt['max_r_length'], transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors+(self.opt['max_count']-length) * [[pad]*self.opt['max_c_length']], vec_lengths+[0]*(self.opt['max_count']-length), length
        else:
            contexts_com=[]
            for sen in contexts[-self.opt['max_count']:-1]:
                contexts_com.extend(sen)
                contexts_com.append('_split_')
            contexts_com.extend(contexts[-1])
            vec,v_l,concept_mask,dbpedia_mask,reviews_mask = self.padding_w2v(contexts_com, self.opt['max_c_length'], transformer)
            return vec,v_l,concept_mask,dbpedia_mask,reviews_mask,0

    def padding_w2v(self,sentence, max_length, transformer=True,pad=0,end=2,unk=3):
        vector=[]
        concept_mask=[]
        dbpedia_mask=[]
        reviews_mask=[]
        for word in sentence:
            #### vector ####
            if '#' in word:
                vector.append(self.word2index.get(word[1:],unk))
            else:
                vector.append(self.word2index.get(word,unk))
            
            #### concept_mask ####
            concept_mask.append(self.key2index.get(word.lower(),0))
            
            #### dbpedia_mask ####
            if '@' in word:
                try:
                    entity = self.id2entity[int(word[1:])]
                    id=self.entity2id[entity]
                except:
                    id=self.entity_max
                dbpedia_mask.append(id)
            else:
                dbpedia_mask.append(self.entity_max)
            
            #### review_mask ####
            if '#' in word:
                reviews_mask.append(1)#self.word2index.get(word[1:],unk))
            else:
                reviews_mask.append(0)#pad)
                
        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)
        reviews_mask.append(0)#pad)

        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length,concept_mask[-max_length:],dbpedia_mask[-max_length:],reviews_mask[:max_length]
            else:
                return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length],reviews_mask[:max_length]
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],\
                   dbpedia_mask+(max_length-len(vector))*[self.entity_max],\
                   reviews_mask+(max_length-len(vector))*[0]
