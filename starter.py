from pathlib import Path

from fastai.vision.data import ImageList
from fastai.vision.learner import cnn_learner
from fastai.vision.models import resnet18
from fastai.vision.transform import get_transforms
from fastai.metrics import accuracy

path = Path('data')

il = ImageList.from_folder(path)
sd = il.split_by_folder(valid='test')
ll = sd.label_from_folder()
ll = ll.transform(get_transforms(), size=224)
data = ll.databunch(bs=32)

data.show_batch()

learn = cnn_learner(data, resnet18, metrics=accuracy)

learn.lr_find()
learn.recorder.plot()


learn.fit_one_cycle(5, 1e-2)

learn.unfreeze()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, slice(1e-4, 1e-3))
