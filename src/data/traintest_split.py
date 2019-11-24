# -*- coding: utf-8 -*-
import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio('../data/processed/', output="../data/images/", seed=1337, ratio=(.85, .15))
print("Done train-test spltting!")
