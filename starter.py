from pathlib import Path

# set callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])
# to save best model (after )
from fastai.callbacks import SaveModelCallback
from fastai.metrics import accuracy
from fastai.train import ClassificationInterpretation
from fastai.vision.data import ImageList
from fastai.vision.learner import cnn_learner
from fastai.vision.models import resnet18
from fastai.vision.transform import ResizeMethod, get_transforms

from radam import RAdam
from ranger import Ranger

# path to the data
path = Path('data')

# create an image list
il = ImageList.from_folder(path)

# cut off bottom of image

# split the data into a train, validation and test set
# be cogniscant of time
sd = il.split_by_folder(valid='test')

# add labels to images
ll = sd.label_from_folder()

# specify data augmentaion
tfms = get_transforms()
ll = ll.transform(tfms, size=256)#, resize_method=ResizeMethod.SQUISH)

# create databunch to pass to model and optimiser
data = ll.databunch(bs=32)

# inspect data
data.show_batch()

# create a learner object with ResNet18 model and adam optimiser (default)
# https://github.com/LiyuanLucasLiu/RAdam
# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer?source=post_page-----2dc83f79a48d----------------------
learn = cnn_learner(data, resnet18, metrics=accuracy, ps=0.5, opt_func=Ranger)
learn = learn.mixup()

# find good learning rate
learn.lr_find()
learn.recorder.plot()

# save best model

learn.fit_one_cycle(10, 1e-2,
learn.save('pre')

preds, y = learn.TTA()
preds, y = learn.get_preds()

accuracy(preds, y)

learn.unfreeze()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(3, slice(1e-5, 1e-4))
