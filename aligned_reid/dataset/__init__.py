import numpy as np
import os
import os.path as osp
ospj = osp.join
ospeu = osp.expanduser

from ..utils.utils import load_pickle
from ..utils.dataset_utils import parse_im_name
from .TrainSet import TrainSet
from .TestSet import TestSet
from .QuerySet import QuerySet


def create_dataset(
    name='market1501',
    part='trainval',
    **kwargs):
  assert name in ['market1501', 'cuhk03', 'duke', 'combined'], \
    "Unsupported Dataset {}".format(name)

  assert part in ['trainval', 'train', 'val', 'test', 'query'], \
    "Unsupported Dataset Part {}".format(part)

  ########################################
  # Specify Directory and Partition File #
  ########################################

  if name == 'market1501':
    im_dir = ospeu('./dataset/market1501/images')
    partition_file = ospeu('./dataset/market1501/partitions.pkl')

  elif name == 'cuhk03':
    im_type = ['detected', 'labeled'][0]
    im_dir = ospeu(ospj('~/Dataset/cuhk03', im_type, 'images'))
    partition_file = ospeu(ospj('~/Dataset/cuhk03', im_type, 'partitions.pkl'))

  elif name == 'duke':
    im_dir = ospeu('~/Dataset/duke/images')
    partition_file = ospeu('~/Dataset/duke/partitions.pkl')

  elif name == 'combined':
    assert part in ['trainval'], \
      "Only trainval part of the combined dataset is available now."
    im_dir = ospeu('~/Dataset/market1501_cuhk03_duke/trainval_images')
    partition_file = ospeu('~/Dataset/market1501_cuhk03_duke/partitions.pkl')

  ##################
  # Create Dataset #
  ##################

  # Use standard Market1501 CMC settings for all datasets here.
  cmc_kwargs = dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)

  partitions = load_pickle(partition_file)
  if not part == 'query':
    im_names = partitions['{}_im_names'.format(part)]

  if part == 'trainval':
    ids2labels = partitions['trainval_ids2labels']

    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)

  elif part == 'train':
    ids2labels = partitions['train_ids2labels']

    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)

  elif part == 'val':
    marks = partitions['val_marks']
    kwargs.update(cmc_kwargs)

    ret_set = TestSet(
      im_dir=im_dir,
      im_names=im_names,
      marks=marks,
      **kwargs)

  elif part == 'test':
    marks = partitions['test_marks']
    kwargs.update(cmc_kwargs)

    ret_set = TestSet(
      im_dir=im_dir,
      im_names=im_names,
      marks=marks,
      **kwargs)

  elif part == 'query':
    marks = partitions['test_marks']
    kwargs.update(cmc_kwargs)
    gal_dir = input('Enter gallery directory: ')
    if not osp.isdir(gal_dir):
      print('Invalid directory.')
      return
    im_names = [f for f in os.listdir(gal_dir) if osp.isfile(osp.join(gal_dir, f))]
    if len(im_names) == 0:
      print('Directory is empty.')
      return
    query_im = input('Enter query path: ')
    if query_im == '':
      print('Invalid path.')
      return
    elif not osp.exists('{}'.format(query_im)):
      print('File \'{}\' not found.'.format(query_im))
      return
      
    ret_set = QuerySet(
      gal_dir=gal_dir,
      gal_names=im_names,
      query_im=query_im,
      **kwargs)

  if part in ['trainval', 'train']:
    num_ids = len(ids2labels)
  elif part in ['val', 'test']:
    ids = [parse_im_name(n, 'id') for n in im_names]
    num_ids = len(list(set(ids)))
    num_query = np.sum(np.array(marks) == 0)
    num_gallery = np.sum(np.array(marks) == 1)
    num_multi_query = np.sum(np.array(marks) == 2)

  # Print dataset information
  print(('-' * 40))
  print(('{} {} set'.format(name, part)))
  print(('-' * 40))

  try:
    print(('NO. Images: {}'.format(len(im_names))))
    print(('NO. IDs: {}'.format(num_ids)))
    print(('NO. Query Images: {}'.format(num_query)))
    print(('NO. Gallery Images: {}'.format(num_gallery)))
    print(('NO. Multi-query Images: {}'.format(num_multi_query)))
  except:
    pass

  print(('-' * 40))

  return ret_set
