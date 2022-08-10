"""Train with optional Global Distance, Local Distance, Identification Loss."""

from pickle import FALSE
import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

from subprocess import run
import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import gradio as gr

from aligned_reid.dataset import create_dataset
from aligned_reid.model.Model import Model
from aligned_reid.model.TripletLoss import TripletLoss
from aligned_reid.model.loss import global_loss
from aligned_reid.model.loss import local_loss

from aligned_reid.utils.utils import time_str
from aligned_reid.utils.utils import str2bool
from aligned_reid.utils.utils import tight_float_str as tfs
from aligned_reid.utils.utils import may_set_mode
from aligned_reid.utils.utils import load_state_dict
from aligned_reid.utils.utils import load_ckpt
from aligned_reid.utils.utils import save_ckpt
from aligned_reid.utils.utils import set_devices
from aligned_reid.utils.utils import AverageMeter
from aligned_reid.utils.utils import to_scalar
from aligned_reid.utils.utils import ReDirectSTD
from aligned_reid.utils.utils import set_seed
from aligned_reid.utils.utils import adjust_lr_exp
from aligned_reid.utils.utils import adjust_lr_staircase


class Config(object):
  def __init__(self, args=None):

    if args == None:
      print("Invalid Arguments.")
      return

    # gpu ids
    self.sys_device_ids = args["sys_device_ids"]

    if args["set_seed"]:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args["run"]

    ###########
    # Dataset #
    ###########

    # If you want to exactly reproduce the result in training, you have to set
    # num of threads to 1.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args["dataset"]
    self.trainset_part = args["trainset_part"]

    # Image Processing

    # Just for training set
    self.crop_prob = args["crop_prob"]
    self.crop_ratio = args["crop_ratio"]
    self.resize_h_w = args["resize_h_w"]

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]

    self.ids_per_batch = args["ids_per_batch"]
    self.ims_per_id = args["ims_per_id"]
    self.train_final_batch = False
    self.train_mirror_type = ['random', 'always', None][0]
    self.train_shuffle = True

    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_mirror_type = ['random', 'always', None][2]
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      ids_per_batch=self.ids_per_batch,
      ims_per_id=self.ims_per_id,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.query_set_kwargs = dict(
      part='query',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.query_set_kwargs.update(dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    self.local_dist_own_hard_sample = args["local_dist_own_hard_sample"]

    self.normalize_feature = args["normalize_feature"]

    self.local_conv_out_channels = 128
    self.global_margin = args["global_margin"]
    self.local_margin = args["local_margin"]

    # Identification Loss weight
    self.id_loss_weight = args["id_loss_weight"]

    # global loss weight
    self.g_loss_weight = args["g_loss_weight"]
    # local loss weight
    self.l_loss_weight = args["l_loss_weight"]

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005

    # Initial learning rate
    self.base_lr = args["base_lr"]
    self.lr_decay_type = args["lr_decay_type"]
    self.exp_decay_at_epoch = args["exp_decay_at_epoch"]
    self.staircase_decay_at_epochs = args["staircase_decay_at_epochs"]
    self.staircase_decay_multiply_factor = args["staircase_decay_multiply_factor"]
    # Number of epochs to train
    self.total_epochs = args["total_epochs"]

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.log_steps = 1e10

    # Only test and without training.
    self.only_test = args["only_test"]

    self.resume = args["resume"]

    # User input query
    self.query_mode = args["query_mode"]

    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args["log_to_file"]

    # The root dir of logs.
    if args["exp_dir"] == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.dataset),
        #
        ('nf_' if self.normalize_feature else 'not_nf_') +
        ('ohs_' if self.local_dist_own_hard_sample else 'not_ohs_') +
        'gm_{}_'.format(tfs(self.global_margin)) +
        'lm_{}_'.format(tfs(self.local_margin)) +
        'glw_{}_'.format(tfs(self.g_loss_weight)) +
        'llw_{}_'.format(tfs(self.l_loss_weight)) +
        'idlw_{}_'.format(tfs(self.id_loss_weight)) +
        'lr_{}_'.format(tfs(self.base_lr)) +
        '{}_'.format(self.lr_decay_type) +
        ('decay_at_{}_'.format(self.exp_decay_at_epoch)
         if self.lr_decay_type == 'exp'
         else 'decay_at_{}_factor_{}_'.format(
          '_'.join([str(e) for e in args["staircase_decay_at_epochs"]]),
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
        #
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args["exp_dir"]

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str())).replace(':', '-')
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str())).replace(':', '-')

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args["model_weight_file"]


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    global_feat, local_feat = self.model(ims)[:2]
    global_feat = global_feat.data.cpu().numpy()
    local_feat = local_feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return global_feat, local_feat


def main(cfg):

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Lazily create SummaryWriter
  writer = None

  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print(('-' * 60))
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print(('-' * 60))

  ###########
  # Dataset #
  ###########

  train_set = create_dataset(**cfg.train_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.dataset)

  ###########
  # Models  #
  ###########

  model = Model(local_conv_out_channels=cfg.local_conv_out_channels,
                num_classes=len(train_set.ids2labels))
  # Model wrapper
  model_w = DataParallel(model)

  #############################
  # Criteria and Optimizers   #
  #############################

  id_criterion = nn.CrossEntropyLoss()
  g_tri_loss = TripletLoss(margin=cfg.global_margin)
  l_tri_loss = TripletLoss(margin=cfg.local_margin)

  optimizer = optim.Adam(model.parameters(),
                         lr=cfg.base_lr,
                         weight_decay=cfg.weight_decay)

  # Bind them together just to save some codes in the following usage.
  modules_optims = [model, optimizer]

  ################################
  # May Resume Models and Optims #
  ################################

  if cfg.resume:
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  TMO(modules_optims)

  #########
  # Query #
  #########

  def query():
    if cfg.model_weight_file != '':
      map_location = (lambda storage, loc: storage)
      sd = torch.load(cfg.model_weight_file, map_location=map_location)
      load_state_dict(model, sd)
      print(('Loaded model weights from {}'.format(cfg.model_weight_file)))
    else:
      load_ckpt(modules_optims, cfg.ckpt_file)
      # print('Model weights not loaded. Please use the model_weight_file arguments.')
      # return
    
    query_set = create_dataset(**cfg.query_set_kwargs)

    use_local_distance = (cfg.l_loss_weight > 0) \
                         and cfg.local_dist_own_hard_sample

    query_set.set_feat_func(ExtractFeature(model_w, TVT))
    print(('\n=========> Probe query on gallery <=========\n'))
    return query_set.probe(
      normalize_feat=cfg.normalize_feature,
      use_local_distance=use_local_distance,)

  if cfg.query_mode:
    results = query()
    imgnames = [osp.basename(name) for name in results[:,:1].flatten()]
    imgmaxlen = len(max(imgnames, key = len))
    print('{:<6} {:<{}} {}'.format('Rank', 'Image Name', imgmaxlen+2, 'Distance'))
    for idx, result in enumerate(results):
      print('{:<6} {:<{}} {}'.format(idx + 1, osp.basename(result[0]), imgmaxlen+2, result[1]))
    return

  ########
  # Test #
  ########

  def test(load_model_weight=False):
    if load_model_weight:
      if cfg.model_weight_file != '':
        map_location = (lambda storage, loc: storage)
        sd = torch.load(cfg.model_weight_file, map_location=map_location)
        load_state_dict(model, sd)
        print(('Loaded model weights from {}'.format(cfg.model_weight_file)))
      else:
        load_ckpt(modules_optims, cfg.ckpt_file)

    use_local_distance = (cfg.l_loss_weight > 0) \
                         and cfg.local_dist_own_hard_sample

    for test_set, name in zip(test_sets, test_set_names):
      test_set.set_feat_func(ExtractFeature(model_w, TVT))
      print(('\n=========> Test on dataset: {} <=========\n'.format(name)))
      return test_set.eval(
        normalize_feat=cfg.normalize_feature,
        use_local_distance=use_local_distance)

  if cfg.only_test:
    return test(load_model_weight=True)

  ############
  # Training #
  ############

  start_ep = resume_ep if cfg.resume else 0
  for ep in range(start_ep, cfg.total_epochs):

    # Adjust Learning Rate
    if cfg.lr_decay_type == 'exp':
      adjust_lr_exp(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.total_epochs,
        cfg.exp_decay_at_epoch)
    else:
      adjust_lr_staircase(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.staircase_decay_at_epochs,
        cfg.staircase_decay_multiply_factor)

    may_set_mode(modules_optims, 'train')

    g_prec_meter = AverageMeter()
    g_m_meter = AverageMeter()
    g_dist_ap_meter = AverageMeter()
    g_dist_an_meter = AverageMeter()
    g_loss_meter = AverageMeter()

    l_prec_meter = AverageMeter()
    l_m_meter = AverageMeter()
    l_dist_ap_meter = AverageMeter()
    l_dist_an_meter = AverageMeter()
    l_loss_meter = AverageMeter()

    id_loss_meter = AverageMeter()

    loss_meter = AverageMeter()

    ep_st = time.time()
    step = 0
    epoch_done = False
    while not epoch_done:

      step += 1
      step_st = time.time()

      ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()

      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      labels_t = TVT(torch.from_numpy(labels).long())
      labels_var = Variable(labels_t)

      global_feat, local_feat, logits = model_w(ims_var)

      g_loss, p_inds, n_inds, g_dist_ap, g_dist_an, g_dist_mat = global_loss(
        g_tri_loss, global_feat, labels_t,
        normalize_feature=cfg.normalize_feature)

      if cfg.l_loss_weight == 0:
        l_loss = 0
      elif cfg.local_dist_own_hard_sample:
        # Let local distance find its own hard samples.
        l_loss, l_dist_ap, l_dist_an, _ = local_loss(
          l_tri_loss, local_feat, None, None, labels_t,
          normalize_feature=cfg.normalize_feature)
      else:
        l_loss, l_dist_ap, l_dist_an = local_loss(
          l_tri_loss, local_feat, p_inds, n_inds, labels_t,
          normalize_feature=cfg.normalize_feature)

      id_loss = 0
      if cfg.id_loss_weight > 0:
        id_loss = id_criterion(logits, labels_var)

      loss = g_loss * cfg.g_loss_weight \
             + l_loss * cfg.l_loss_weight \
             + id_loss * cfg.id_loss_weight

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      ############
      # Step Log #
      ############

      # precision
      g_prec = (g_dist_an > g_dist_ap).data.float().mean()
      # the proportion of triplets that satisfy margin
      g_m = (g_dist_an > g_dist_ap + cfg.global_margin).data.float().mean()
      g_d_ap = g_dist_ap.data.mean()
      g_d_an = g_dist_an.data.mean()

      g_prec_meter.update(g_prec)
      g_m_meter.update(g_m)
      g_dist_ap_meter.update(g_d_ap)
      g_dist_an_meter.update(g_d_an)
      g_loss_meter.update(to_scalar(g_loss))

      if cfg.l_loss_weight > 0:
        # precision
        l_prec = (l_dist_an > l_dist_ap).data.float().mean()
        # the proportion of triplets that satisfy margin
        l_m = (l_dist_an > l_dist_ap + cfg.local_margin).data.float().mean()
        l_d_ap = l_dist_ap.data.mean()
        l_d_an = l_dist_an.data.mean()

        l_prec_meter.update(l_prec)
        l_m_meter.update(l_m)
        l_dist_ap_meter.update(l_d_ap)
        l_dist_an_meter.update(l_d_an)
        l_loss_meter.update(to_scalar(l_loss))

      if cfg.id_loss_weight > 0:
        id_loss_meter.update(to_scalar(id_loss))

      loss_meter.update(to_scalar(loss))

      if step % cfg.log_steps == 0:
        time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
          step, ep + 1, time.time() - step_st, )

        if cfg.g_loss_weight > 0:
          g_log = (', gp {:.2%}, gm {:.2%}, '
                   'gd_ap {:.4f}, gd_an {:.4f}, '
                   'gL {:.4f}'.format(
            g_prec_meter.val, g_m_meter.val,
            g_dist_ap_meter.val, g_dist_an_meter.val,
            g_loss_meter.val, ))
        else:
          g_log = ''

        if cfg.l_loss_weight > 0:
          l_log = (', lp {:.2%}, lm {:.2%}, '
                   'ld_ap {:.4f}, ld_an {:.4f}, '
                   'lL {:.4f}'.format(
            l_prec_meter.val, l_m_meter.val,
            l_dist_ap_meter.val, l_dist_an_meter.val,
            l_loss_meter.val, ))
        else:
          l_log = ''

        if cfg.id_loss_weight > 0:
          id_log = (', idL {:.4f}'.format(id_loss_meter.val))
        else:
          id_log = ''

        total_loss_log = ', loss {:.4f}'.format(loss_meter.val)

        log = time_log + \
              g_log + l_log + id_log + \
              total_loss_log
        print(log)

    #############
    # Epoch Log #
    #############

    time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st, )

    if cfg.g_loss_weight > 0:
      g_log = (', gp {:.2%}, gm {:.2%}, '
               'gd_ap {:.4f}, gd_an {:.4f}, '
               'gL {:.4f}'.format(
        g_prec_meter.avg, g_m_meter.avg,
        g_dist_ap_meter.avg, g_dist_an_meter.avg,
        g_loss_meter.avg, ))
    else:
      g_log = ''

    if cfg.l_loss_weight > 0:
      l_log = (', lp {:.2%}, lm {:.2%}, '
               'ld_ap {:.4f}, ld_an {:.4f}, '
               'lL {:.4f}'.format(
        l_prec_meter.avg, l_m_meter.avg,
        l_dist_ap_meter.avg, l_dist_an_meter.avg,
        l_loss_meter.avg, ))
    else:
      l_log = ''

    if cfg.id_loss_weight > 0:
      id_log = (', idL {:.4f}'.format(id_loss_meter.avg))
    else:
      id_log = ''

    total_loss_log = ', loss {:.4f}'.format(loss_meter.avg)

    log = time_log + \
          g_log + l_log + id_log + \
          total_loss_log
    print(log)

    # Log to TensorBoard

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'loss',
        dict(global_loss=g_loss_meter.avg,
             local_loss=l_loss_meter.avg,
             id_loss=id_loss_meter.avg,
             loss=loss_meter.avg, ),
        ep)
      writer.add_scalars(
        'tri_precision',
        dict(global_precision=g_prec_meter.avg,
             local_precision=l_prec_meter.avg, ),
        ep)
      writer.add_scalars(
        'satisfy_margin',
        dict(global_satisfy_margin=g_m_meter.avg,
             local_satisfy_margin=l_m_meter.avg, ),
        ep)
      writer.add_scalars(
        'global_dist',
        dict(global_dist_ap=g_dist_ap_meter.avg,
             global_dist_an=g_dist_an_meter.avg, ),
        ep)
      writer.add_scalars(
        'local_dist',
        dict(local_dist_ap=l_dist_ap_meter.avg,
             local_dist_an=l_dist_an_meter.avg, ),
        ep)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)

  ########
  # Test #
  ########

  return test(load_model_weight=False)


def start_train(train_sys_device_ids, train_run, train_set_seed,
                train_dataset, train_trainset_part, train_resize_h_w, 
                train_crop_prob, train_crop_ratio, train_ids_per_batch,
                train_ims_per_id, train_log_to_file, train_normalize_feature,
                train_local_dist_own_hard_sample, train_global_margin,
                train_local_margin, train_g_loss_weight, train_l_loss_weight, 
                train_id_loss_weight, train_resume, train_exp_dir, train_base_lr, 
                train_lr_decay_type, train_exp_decay_at_epoch, 
                train_staircase_decay_at_epochs, 
                train_staircase_decay_multiply_factor, 
                train_total_epochs):
    train_args = {
      "sys_device_ids": eval(train_sys_device_ids),
      "run": int(train_run),
      "set_seed": str2bool(train_set_seed),
      "dataset": train_dataset,
      "trainset_part": train_trainset_part,
      "resize_h_w": eval(train_resize_h_w),
      "crop_prob": train_crop_prob,
      "crop_ratio": train_crop_ratio,
      "ids_per_batch": int(train_ids_per_batch),
      "ims_per_id": int(train_ims_per_id),
      "log_to_file": str2bool(train_log_to_file),
      "normalize_feature": str2bool(train_normalize_feature),
      "local_dist_own_hard_sample": str2bool(train_local_dist_own_hard_sample),
      "global_margin": train_global_margin,
      "local_margin": train_local_margin,
      "g_loss_weight": train_g_loss_weight,
      "l_loss_weight": train_l_loss_weight,
      "id_loss_weight": train_id_loss_weight,
      "only_test": False,
      "resume": str2bool(train_resume),
      "exp_dir": train_exp_dir,
      "model_weight_file": '',
      "base_lr": train_base_lr,
      "lr_decay_type": train_lr_decay_type,
      "exp_decay_at_epoch": int(train_exp_decay_at_epoch),
      "staircase_decay_at_epochs": eval(train_staircase_decay_at_epochs),
      "staircase_decay_multiply_factor": train_staircase_decay_multiply_factor,
      "total_epochs": int(train_total_epochs),
      "query_mode": False,
    }

    cfg = Config(train_args)
    mAP, cmc_scores, mq_mAP, mq_cmc_scores = main(cfg)

    scores = {
      "mAP": mAP,
      "CMC1": cmc_scores[0],
      "CMC5": cmc_scores[4],
      "CMC10": cmc_scores[9]
    }

    return scores


def start_test(test_sys_device_ids, test_run, test_set_seed,
              test_dataset, test_trainset_part, test_resize_h_w, 
              test_crop_prob, test_crop_ratio, test_ids_per_batch,
              test_ims_per_id, test_log_to_file, test_normalize_feature,
              test_local_dist_own_hard_sample, test_global_margin,
              test_local_margin, test_g_loss_weight, test_l_loss_weight, 
              test_id_loss_weight, test_resume, test_exp_dir, 
              test_model_weight_file):
    test_args = {
      "sys_device_ids": eval(test_sys_device_ids),
      "run": int(test_run),
      "set_seed": str2bool(test_set_seed),
      "dataset": test_dataset,
      "trainset_part": test_trainset_part,
      "resize_h_w": eval(test_resize_h_w),
      "crop_prob": test_crop_prob,
      "crop_ratio": test_crop_ratio,
      "ids_per_batch": int(test_ids_per_batch),
      "ims_per_id": int(test_ims_per_id),
      "log_to_file": str2bool(test_log_to_file),
      "normalize_feature": str2bool(test_normalize_feature),
      "local_dist_own_hard_sample": str2bool(test_local_dist_own_hard_sample),
      "global_margin": test_global_margin,
      "local_margin": test_local_margin,
      "g_loss_weight": test_g_loss_weight,
      "l_loss_weight": test_l_loss_weight,
      "id_loss_weight": test_id_loss_weight,
      "only_test": True,
      "resume": str2bool(test_resume),
      "exp_dir": test_exp_dir,
      "model_weight_file": test_model_weight_file,
      "base_lr": 1e-3,
      "lr_decay_type": 'staircase',
      "exp_decay_at_epoch": 0,
      "staircase_decay_at_epochs": 0,
      "staircase_decay_multiply_factor": 0,
      "total_epochs": 0,
      "query_mode": False,
    }

    cfg = Config(test_args)
    mAP, cmc_scores, mq_mAP, mq_cmc_scores = main(cfg)

    scores = {
      "mAP": mAP,
      "CMC1": cmc_scores[0],
      "CMC5": cmc_scores[4],
      "CMC10": cmc_scores[9]
    }

    return scores

if __name__ == '__main__':
  with gr.Blocks() as app:
    gr.Markdown("Person Re-identification using AlignedReID Method")
    with gr.Tabs():
      with gr.TabItem("Training & Testing"):
        with gr.Row():
          with gr.Column():
            with gr.Row():
              train_sys_device_ids = gr.Textbox(value='(0,)', label='System Device IDs')
              train_run = gr.Number(value=1, label='Number of Run(s)')
            with gr.Row():
              train_set_seed = gr.Radio(choices=['True', 'False'], value='False', label='Use seed to randomize dataset distribution')
              train_dataset = gr.Radio(choices=['market1501', 'cuhk03', 'duke', 'combined'], value='market1501', label='Person Re-ID dataset to use')
            with gr.Row():
              train_trainset_part = gr.Radio(choices=['trainval', 'train'], value='trainval', label='Trainset part to use')
              
              train_resize_h_w = gr.Textbox(value='(224,224)', label='Resize images height and width to (h,w)')
            with gr.Row():
              train_crop_prob = gr.Number(value=0, label='The probability of each image to go through cropping')
              train_crop_ratio = gr.Number(value=1, label='Cropping ratio (if == 1.0, no cropping)')
            with gr.Row():
              train_ids_per_batch = gr.Number(value=16, label='Number of IDs per batch')
              train_ims_per_id = gr.Number(value=4, label='Number of imagess per batch')

            with gr.Row():
              train_log_to_file = gr.Radio(choices=['True', 'False'], value='True', label='Save logs to file')
              train_normalize_feature = gr.Radio(choices=['True', 'False'], value='False', label='Normalize feature')
              train_local_dist_own_hard_sample = gr.Radio(choices=['True', 'False'], value='False', label='Use own hard sample for local distance')

            with gr.Row():
              train_global_margin = gr.Number(value=0.5, label='Global margin for triplet hard loss function')
              train_local_margin = gr.Number(value=0.5, label='Local margin for triplet hard loss function')
            with gr.Row():
              train_g_loss_weight = gr.Number(value=0.5, label='Global loss weight')
              train_l_loss_weight = gr.Number(value=0.5, label='Local loss weight')
              train_id_loss_weight = gr.Number(value=0., label='Identity loss weight')
            
            with gr.Row():
              train_resume = gr.Radio(choices=['True', 'False'], value='False', label='Resume previous runs')
              train_exp_dir = gr.Textbox(label='Experiment directory (uses root folder if left empty)')
            
            with gr.Row():
              train_base_lr = gr.Number(value=1e-3, label='Base learning rate')
              train_lr_decay_type = gr.Radio(choices=['exp', 'staircase'], value='staircase', label='Learning rate decay type')
            with gr.Row():
              train_exp_decay_at_epoch = gr.Number(value=76, label='Epoch in which exponential decay happen (if used)')
              train_staircase_decay_at_epochs = gr.Textbox(value='(80,160)', label='Epochs in which staircase decay happen (if used)')
            with gr.Row():
              train_staircase_decay_multiply_factor = gr.Number(value=0.1, label='Staircase decay multiply factor')
              train_total_epochs = gr.Number(value=240, label='Total epochs')
            train_button = gr.Button("Start").style(full_width=True)

          with gr.Column():
            gr.Markdown("Training & Testing Scores Output")
            scores = gr.Label(num_top_classes=4, label="Evaluation Metrics Scores")

      with gr.TabItem("Testing Only"):
        with gr.Row():
          with gr.Column():
            with gr.Row():
              test_sys_device_ids = gr.Textbox(value='(0,)', label='System Device IDs')
              test_run = gr.Number(value=1, label='Number of Run(s)')
            with gr.Row():
              test_set_seed = gr.Radio(choices=['True', 'False'], value='False', label='Use seed to randomize dataset distribution')
              test_dataset = gr.Radio(choices=['market1501', 'cuhk03', 'duke', 'combined'], value='market1501', label='Person Re-ID dataset to use')
            with gr.Row():
              test_trainset_part = gr.Radio(choices=['trainval', 'train'], value='trainval', label='Trainset part to use')
              
              test_resize_h_w = gr.Textbox(value='(224,224)', label='Resize images height and width to (h,w)')
            with gr.Row():
              test_crop_prob = gr.Number(value=0, label='The probability of each image to go through cropping')
              test_crop_ratio = gr.Number(value=1, label='Cropping ratio (if == 1.0, no cropping)')
            with gr.Row():
              test_ids_per_batch = gr.Number(value=16, label='Number of IDs per batch')
              test_ims_per_id = gr.Number(value=4, label='Number of imagess per batch')

            with gr.Row():
              test_log_to_file = gr.Radio(choices=['True', 'False'], value='True', label='Save logs to file')
              test_normalize_feature = gr.Radio(choices=['True', 'False'], value='False', label='Normalize feature')
              test_local_dist_own_hard_sample = gr.Radio(choices=['True', 'False'], value='False', label='Use own hard sample for local distance')

            with gr.Row():
              test_global_margin = gr.Number(value=0.5, label='Global margin for triplet hard loss function')
              test_local_margin = gr.Number(value=0.5, label='Local margin for triplet hard loss function')
            with gr.Row():
              test_g_loss_weight = gr.Number(value=0.5, label='Global loss weight')
              test_l_loss_weight = gr.Number(value=0.5, label='Local loss weight')
              test_id_loss_weight = gr.Number(value=0., label='Identity loss weight')
            
            with gr.Row():
              test_resume = gr.Radio(choices=['True', 'False'], value='False', label='Resume previous runs')
              test_exp_dir = gr.Textbox(label='Experiment directory (uses root folder if left empty)')
              test_model_weight_file = gr.Textbox(label='Model weight file path (uses available checkpoint if left empty)')
            
            test_button = gr.Button("Start").style(full_width=True)

          with gr.Column():
            gr.Markdown("Training & Testing Scores Output")
            test_scores = gr.Label(num_top_classes=4, label="Evaluation Metrics Scores")
          
      with gr.TabItem("Query Mode"):
          with gr.Row():
              image_input = gr.Image()
              image_output = gr.Image()
          image_button = gr.Button("Flip")

    train_button.click(start_train, 
        inputs=[
            train_sys_device_ids, train_run, train_set_seed,
            train_dataset, train_trainset_part, train_resize_h_w, 
            train_crop_prob, train_crop_ratio, train_ids_per_batch,
            train_ims_per_id, train_log_to_file, train_normalize_feature,
            train_local_dist_own_hard_sample, train_global_margin,
            train_local_margin, train_g_loss_weight, train_l_loss_weight, 
            train_id_loss_weight, train_resume, train_exp_dir, train_base_lr, 
            train_lr_decay_type, train_exp_decay_at_epoch, 
            train_staircase_decay_at_epochs, 
            train_staircase_decay_multiply_factor, 
            train_total_epochs
          ], 
        outputs=scores)
    test_button.click(start_test, 
        inputs=[
            test_sys_device_ids, test_run, test_set_seed,
            test_dataset, test_trainset_part, test_resize_h_w, 
            test_crop_prob, test_crop_ratio, test_ids_per_batch,
            test_ims_per_id, test_log_to_file, test_normalize_feature,
            test_local_dist_own_hard_sample, test_global_margin,
            test_local_margin, test_g_loss_weight, test_l_loss_weight, 
            test_id_loss_weight, test_resume, test_exp_dir, 
            test_model_weight_file
          ], 
        outputs=test_scores)
    image_button.click(start_test, inputs=image_input, outputs=image_output)

  app.launch(enable_queue=True)
