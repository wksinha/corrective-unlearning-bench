import torch, methods, resnet, timm
import numpy as np
from os import makedirs
from os.path import exists
from torch.utils.data.sampler import SubsetRandomSampler
from opts import parse_args 
from utils import seed_everything, SubsetSequentialSampler, get_targeted_classes  
from datasets import load_dataset, DatasetWrapper, manip_dataset, get_deletion_set
from torchvision import transforms
import matplotlib.pyplot as plt
import random

def filter_dog_cat(dataset, cat_fraction, is_train):
    dog_cat_indices = []
    for i in range(len(dataset)):
        if dataset.targets[i] == 5:
            dog_cat_indices.append(i)
        elif dataset.targets[i] == 3:
            if is_train:
                if random.random() < cat_fraction:
                    dog_cat_indices.append(i)
            else:
                dog_cat_indices.append(i)

    return dog_cat_indices

class RetainDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, indices_to_retain):
        self.original_dataset = original_dataset
        self.indices_to_retain = indices_to_retain

    def __len__(self):
        return len(self.indices_to_retain)

    def __getitem__(self, idx):
        original_idx = self.indices_to_retain[idx]
        return self.original_dataset[original_idx]


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_everything(seed=0)
    # assert(torch.cuda.is_available())
    opt = parse_args()
    print('==> Opts: ',opt)

    # Get model
    if opt.model == 'vitb16':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=opt.num_classes)
    else:
        model = getattr(resnet, opt.model)(opt.num_classes)

    # Get dataloaders done
    train_set, train_noaug_set, test_set, train_labels, max_val = load_dataset(dataset=opt.dataset, root=opt.data_dir)
    train_indices = filter_dog_cat(train_set, opt.cat_fraction, is_train=True)
    train_noaug_indices = filter_dog_cat(train_noaug_set, opt.cat_fraction, is_train=True)
    test_indices = filter_dog_cat(test_set, opt.cat_fraction, is_train=False)
    train_set = RetainDataset(train_set, train_indices)
    train_noaug_set = RetainDataset(train_noaug_set, train_noaug_indices)
    print(len(train_noaug_set))
    train_labels = RetainDataset(train_labels, train_indices)
    test_set = RetainDataset(test_set, test_indices)
    # for x, y in train_set:
    #     assert(y in [3, 5])
    # for x, y in train_noaug_set:
    #     assert(y in [3, 5])
    # for x, y in test_set:
    #     assert(y in [3, 5])

    assert(opt.binary_poison_ratio * opt.forget_set_size <= len(train_set) - 5000)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    manip_dict, manip_idx, untouched_idx = manip_dataset(dataset='CatDogCIFAR10', train_labels=train_labels, method=opt.dataset_method, manip_set_size=opt.forget_set_size, save_dir=opt.save_dir, cat_fraction=opt.cat_fraction, binary_poison_ratio=opt.binary_poison_ratio)
    print('==> Loaded the dataset!')

    wtrain_noaug_cleanL_set = DatasetWrapper(train_noaug_set, manip_dict, mode='test')
    train_test_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(untouched_idx), num_workers=4, pin_memory=True)
    manip_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(manip_idx), num_workers=4, pin_memory=True)
    eval_loaders = {}
    # for x, y in wtrain_noaug_cleanL_set:
    #     assert(y in [3, 5])

    if opt.dataset_method == 'poisoning':
        corrupt_val = np.array(max_val)
        corrupt_size = opt.patch_size
        wtrain_noaug_adv_cleanL_set = DatasetWrapper(train_noaug_set, manip_dict, mode='test_adversarial', corrupt_val=corrupt_val, corrupt_size=corrupt_size)
        adversarial_train_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # for x, y in wtrain_noaug_adv_cleanL_set:
        #     assert(y in [3, 5])
        untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(untouched_idx), num_workers=4, pin_memory=True)
        manip_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(manip_idx), num_workers=4, pin_memory=True)
        wtest_adv_cleanL_set = DatasetWrapper(test_set, manip_dict, mode='test_adversarial', corrupt_val=corrupt_val, corrupt_size=corrupt_size)
        # for x, y in wtest_adv_cleanL_set:
        #     assert(y in [3, 5])

        adversarial_test_loader = torch.utils.data.DataLoader(wtest_adv_cleanL_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        eval_loaders['adv_test'] = adversarial_test_loader
    else:
        adversarial_train_loader, adversarial_test_loader, corrupt_val, corrupt_size = None, None, None, None

    eval_loaders['manip'] = manip_noaug_cleanL_loader
    if opt.dataset_method == 'labeltargeted':
        classes = get_targeted_classes(opt.dataset)
        indices = []
        for batch_idx, (data, target) in enumerate(test_loader):
            matching_indices = (target == classes[0]) | (target == classes[1])
            absolute_indices = batch_idx * test_loader.batch_size + torch.where(matching_indices)[0]
            indices.extend(absolute_indices.tolist())
        eval_loaders['unseen_forget'] = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(indices), num_workers=4, pin_memory=True)

    wtrain_manip_set = DatasetWrapper(train_set, manip_dict, mode='pretrain', corrupt_val=corrupt_val, corrupt_size=corrupt_size)
    # for row in wtrain_manip_set:
    #     assert(row[1] in [3, 5])
    pretrain_loader = torch.utils.data.DataLoader(wtrain_manip_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Stage 1: Pretraining
    opt.pretrain_file_prefix = opt.save_dir+'/'+opt.dataset+'_'+opt.model+'_'+opt.dataset_method+'_'+str(opt.forget_set_size)+'_'+str(opt.patch_size)+'_'+str(opt.pretrain_iters)+'_'+str(opt.pretrain_lr)+'_'+str(opt.cat_fraction)+'_'+str(opt.binary_poison_ratio)
    if not exists(opt.pretrain_file_prefix):makedirs(opt.pretrain_file_prefix)
    print(opt.deletion_size, "DELEEE")

    if not exists(opt.pretrain_file_prefix + '/Naive_pretrainmodel/model.pth'):
        opt.max_lr, opt.train_iters, expname, unlearn_method = opt.pretrain_lr, opt.pretrain_iters, opt.exp_name, opt.unlearn_method
        
        #We now actually pretrain by calling unlearn(), misnomer
        opt.unlearn_method, opt.exp_name = 'Naive', 'pretrainmodel'
        method = getattr(methods, opt.unlearn_method)(opt=opt, model=model)
        method.unlearn(train_loader=pretrain_loader, test_loader=test_loader)
        method.compute_and_save_results(train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader)
        opt.exp_name, opt.unlearn_method = expname, unlearn_method  
    else:
        print('==> Loading the pretrained model!')
        model.load_state_dict(torch.load(opt.pretrain_file_prefix + '/Naive_pretrainmodel/model.pth'))
        model.to(opt.device)
        print('==> Loaded the pretrained model!')

    #deletion set
    if opt.deletion_size is None:
        opt.deletion_size = opt.forget_set_size
    forget_idx, retain_idx = get_deletion_set(opt.forget_set_size, opt.deletion_size, manip_dict, train_size=len(train_labels), dataset=opt.dataset, method=opt.dataset_method, cat_fraction=opt.cat_fraction, binary_poison_ratio=opt.binary_poison_ratio, save_dir=opt.save_dir)    
    opt.max_lr, opt.train_iters = opt.unlearn_lr, opt.unlearn_iters 
    if opt.deletion_size != len(manip_dict):
        delete_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(forget_idx), num_workers=4, pin_memory=True)
        if opt.dataset_method == 'poisoning':
            delete_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(forget_idx), num_workers=4, pin_memory=True)
        eval_loaders['delete'] = delete_noaug_cleanL_loader
        
    # Stage 2: Unlearning
    method = getattr(methods, 'ApplyK')(opt=opt, model=model) if opt.unlearn_method in ['EU', 'CF'] else getattr(methods, opt.unlearn_method)(opt=opt, model=model)

    wtrain_delete_set = DatasetWrapper(train_set, manip_dict, mode='pretrain', corrupt_val=corrupt_val, corrupt_size=corrupt_size, delete_idx=forget_idx)
    # Get the dataloaders
    retain_loader = torch.utils.data.DataLoader(wtrain_delete_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetRandomSampler(retain_idx), num_workers=4, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(wtrain_delete_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    forget_loader = torch.utils.data.DataLoader(wtrain_delete_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetRandomSampler(forget_idx), num_workers=4, pin_memory=True)

    if opt.unlearn_method in ['Naive', 'EU', 'CF']:
        method.unlearn(train_loader=retain_loader, test_loader=test_loader, eval_loaders=eval_loaders)
    elif opt.unlearn_method in ['BadT']:
        method.unlearn(train_loader=train_loader, test_loader=test_loader, eval_loaders=eval_loaders)
    elif opt.unlearn_method in ['Scrub', 'SSD']:
        method.unlearn(train_loader=retain_loader, test_loader=test_loader, forget_loader=forget_loader, eval_loaders=eval_loaders)
    
    method.compute_and_save_results(train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader)
    print('==> Experiment completed! Exiting..')
