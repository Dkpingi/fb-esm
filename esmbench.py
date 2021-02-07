from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import esm
from esm.data import FastaBatchedDataset

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

proteindata = FastaBatchedDataset.from_file("../scratch/proteinnet7.fa")

valloader = DataLoader(proteindata, batch_size = 2, collate_fn = batch_converter)

loss = torch.nn.CrossEntropyLoss()

#model.to('cuda')

precision = 0.0
lss = 0.0
explss = 0.0
for i,data in enumerate(valloader):
    with torch.no_grad():
        batch_labels, batch_strs, batch_tokens = data
        #batch_tokens.to('cuda')

        mask_id = np.random.choice(range(batch_tokens.size(1)), size = int(0.15*batch_tokens.size(1)), replace = False)
        keep_id = np.random.choice(mask_id,size = int(0.2*len(mask_id)), replace = False)
        swap_id = np.random.choice(keep_id,size = int(0.5*len(keep_id)), replace = False)
        swap = np.random.choice(range(4,24), size = len(swap_id))

        mask_tokens = torch.clone(batch_tokens)
        mask_tokens[:,mask_id] = 32
        mask_tokens[:,keep_id] = batch_tokens[:,keep_id]
        mask_tokens[:,swap_id] = torch.tensor(swap)
        
        was_masked = torch.zeros_like(mask_tokens)
        was_masked[:,mask_id] = 1
        
        is_relevant = torch.logical_and(batch_tokens >= 4, batch_tokens <= 23)
        
        relevant_masks = torch.logical_and(is_relevant, was_masked)

        mask_tokens = torch.where(relevant_masks, mask_tokens, batch_tokens)
        output = model(mask_tokens)
        predictions = torch.argmax(output['logits'], dim = 2)

        correct = torch.sum(torch.logical_and(predictions == batch_tokens, relevant_masks == 1))
        mask_num = torch.sum(relevant_masks)
        logits = output['logits'][relevant_masks,:]
        lss_ = loss(logits,batch_tokens[relevant_masks]).item()
        lss += lss_
        explss += np.exp(lss_)
        if mask_num != 0:         
            precision += float(correct/mask_num)
        else:
            precision += 1.0
        #if i % 100 == 0:
        print(precision/(i+1))
        print(lss/(i+1))
        print(explss/(i+1))

print(precision/(i+1))

