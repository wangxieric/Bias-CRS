import torch.nn as nn
import json
from bias_crs.model.base import BaseModel
from bias_crs.model.crs.revcore.cross_model import CrossModel

class RevCoreRecModel(BaseModel):
    def __init__(self, opt, device, dpath=None, resource=None):
        super().__init__(opt, device, dpath, resource)
    
    def build_model(self, *args, **kwargs):
        self.model = CrossModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()
        return super().build_model(*args, **kwargs)

    def forward(self, batch, epoch, mode):
        seed_sets = []
        for b in range(self.opt['rec']['batch_size']):
            seed_set = batch['context_entities'][b].nonzeros().view(-1).tolist()
            seed_sets.append(seed_set)
        self.model.train()
        self.zero_grad()
        scores, _, _, rec_loss, _, _, info_db_loss, _ = self.model(batch['context_token'], batch['response'], 
                                            batch['mask_response'],  batch['concept_mask'],
                                            batch['reviews_mask'], seed_sets, batch['items'], batch['concept_vec'], 
                                            batch['db_vec'], batch['entity_vector'], batch['rec'], test=False)
        if epoch < 3:
            joint_loss = info_db_loss
        else:
            joint_loss = rec_loss+0.025*info_db_loss
        # joint_loss.backward()
        # self.optimizer.step()
        loss = joint_loss
        return loss, scores

