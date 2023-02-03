# @Time   : 2020/11/27
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/2
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import torch
from tqdm import tqdm

from bias_crs.data.dataloader.base import BaseDataLoader
from bias_crs.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class KBRDDataLoader(BaseDataLoader):
    """Dataloader for model KBRD.

    Notes:
        You can set the following parameters in config:

        - ``'context_truncate'``: the maximum length of context.
        - ``'response_truncate'``: the maximum length of response.
        - ``'entity_truncate'``: the maximum length of mentioned entities in context.

        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'pad_entity'``

        the above values specify the id of needed special token.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.pad_entity_idx = vocab['pad_entity']
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)

    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                for movie in conv_dict['items']:
                    augment_conv_dict = {'context_entities': conv_dict['context_entities'], 'item': movie, 
                                        'role': conv_dict['role'],
                                        'conv_id': conv_dict['conv_id'],
                                        'user_id': conv_dict['user_id'],
                                        'context_tokens': conv_dict['context_tokens'],
                                        'response': conv_dict['response'],
                                        'context_words': conv_dict['context_words'],
                                        'context_items': conv_dict['context_items']}
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch):
        batch_context_entities = []
        batch_movies = []
        
        # for analysis
        batch_conv_ids = []
        batch_context_roles = []
        batch_context_users = []
        batch_context_items = []
        batch_context_entities = []
        batch_context_words = []
        batch_context_tokens = []
        
        for conv_dict in batch:
            batch_context_entities.append(conv_dict['context_entities'])
            batch_movies.append(conv_dict['item'])
            
            # add addition data for analysis
            batch_conv_ids.append(conv_dict['conv_id'])
            batch_context_roles.append(conv_dict['role'])
            batch_context_users.append(conv_dict['user_id'])
            batch_context_items.append(conv_dict['context_items'])
            # batch_context_entities.append(conv_dict['context_entities'])
            batch_context_words.append(conv_dict['context_words'])
            batch_context_tokens.append(conv_dict['context_tokens'])

        return {
            "context_entities": batch_context_entities,
            "item": torch.tensor(batch_movies, dtype=torch.long),
            'related_data': {"conv_id": batch_conv_ids,
            "roles": batch_context_roles,
            "user_id": batch_context_users,
            "context_item_ids": batch_context_items,
            "entity_ids": batch_context_entities,
            "word_ids": batch_context_words,
            "token_ids": batch_context_tokens}
            }

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        batch_context_tokens = []
        batch_context_entities = []
        batch_response = []
        for conv_dict in batch:
            batch_context_tokens.append(
                truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            batch_context_entities.append(conv_dict['context_entities'])
            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))

        return {
            "context_tokens": padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
            "context_entities": batch_context_entities,
            "response": padded_tensor(batch_response, self.pad_token_idx)
        }

    def policy_batchify(self, *args, **kwargs):
        pass
