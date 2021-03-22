# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:17:26 2020

@author: ramji
"""

""" for testing the dataset file"""
image_files = glob.glob(os.path.join(config.DATA_DIR,"*.png"))
targets_orig = [x.split('/')[-1][:-4] for x in image_files]
targets = [[c for c in t] for t in targets_orig]
target_flat = [c for t in targets for c in t ]

lbl_enc = preprocessing.LabelEncoder()
lbl_enc.fit(target_flat)
targets_enc = [lbl_enc.transform(x) for x in targets]
targets_enc = np.array(targets_enc)
targets_enc = targets_enc + 1

train_imgs,test_imgs,train_targets,test_targets,_,test_targets_orig = model_selection.train_test_split(image_files, targets_enc, targets_orig, test_size=0.1, random_state=42)

train_dataset = dataset.ClassificationDataset(
    image_paths=train_imgs,
    targets=train_targets,
    resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),)
i =train_dataset[0]['images'].numpy()
i.shape