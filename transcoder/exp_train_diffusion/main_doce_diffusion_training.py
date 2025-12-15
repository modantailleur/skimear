import numpy as np
from pathlib import Path
import time
import torch
import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)
from doce.setting import Setting
import utils.util as ut
import diffusion_experiment_manager
import doce
import copy

torch.manual_seed(71)

if torch.cuda.is_available():
    # Set the random seed for GPU (if available)
    torch.cuda.manual_seed(0)

# define the experiment
experiment = doce.Experiment(
  name = "DiffusionExp16042024",
  # name = "DiffusionExp07052024",
  purpose = 'experiment for spectral transcoder',
  author = 'Modan Tailleur',
  address = 'modan.tailleur@ls2n.fr',
)

#########################
#########################
# EXPERIMENT PATH

#general
# EXP_PATH = '../6-diffusionFromThirdo-Data/training_experiments/'

#for jean zay
# EXP_PATH = '/gpfswork/rech/dpr/uml42ji/6-diffusionFromThirdo-Data/training_experiments/'

#for ssd
EXP_PATH = '/media/user/MT-SSD-3/0-PROJETS_INFO/Thèse/6-DiffusionFromThirdo-Data/training_experiments/'

#########################
######################################
# PROJECT TRAIN DATA PATH

#general
PROJECT_DATA_PATH = '../6-diffusionFromThirdo-Data/spectral_data'

#for jean zay
# PROJECT_DATA_PATH = '/gpfswork/rech/dpr/uml42ji/6-diffusionFromThirdo-Data/spectral_data'

#########################
######################################
# DATASETS PATH

#general
PROJECT_DATASET_PATH = '../6-diffusionFromThirdo-Data/datasets'

#for jean zay
# PROJECT_DATASET_PATH = '/gpfswork/rech/dpr/uml42ji/6-diffusionFromThirdo-Data/datasets'

if not os.path.exists(EXP_PATH):
    # Create the directory recursively
    os.makedirs(EXP_PATH)

experiment.set_path('output', EXP_PATH+experiment.name+'/', force=True)
experiment.set_path('pann_output', EXP_PATH+experiment.name+'/pann_output', force=True)
experiment.set_path('audio_output', EXP_PATH+experiment.name+'/audio_output', force=True)
experiment.set_path('mel_output', EXP_PATH+experiment.name+'/mel_output', force=True)
experiment.set_path('logit_output', EXP_PATH+experiment.name+'/logit_output', force=True)
experiment.set_path('duration', EXP_PATH+experiment.name+'/duration/', force=True)
experiment.set_path('model', EXP_PATH+experiment.name+'/model/', force=True)
experiment.set_path('loss', EXP_PATH+experiment.name+'/loss/', force=True)
experiment.set_path('metric', EXP_PATH+experiment.name+'/metric/', force=True)
#by default, export path is in "./export/"
experiment.set_path('export', EXP_PATH+experiment.name+'/export/', force=True)

###################
#transcode 1/3oct to mels using baseline (oracle and pseudo-inverse). Oracle corresponds to audio --> mels --> audio
experiment.add_plan('baseline',
  method = ['oracle', 'pinv', 'random'],
  step = ['eval', 'vocode', 'metric'],
  tho_type = ['fast', 'slow'],
  evalset = ['ljspeech', 'librispeech', 'ljspeech_45dba', 'ljspeech_50dba', 'ljspeech_55dba', 'ljspeech_60dba', 'ljspeech_65dba', 'ljspeech_70dba', 'ljspeech_75dba'],
)

###################
#transcode 1/3oct to mels using transcoder from (Tailleur et al., 2023). Is called CNN in the code.
#link: https://hal.science/hal-04178197v2/file/DCASE_2023___spectral_transcoder_camera_ready_14082023.pdf
#Only used for ENVIRONMENTAL AUDIO (not VOICE)
experiment.add_plan('transcoder',
  method = ['transcoder'],
  step = ['train', 'eval', 'vocode', 'metric'],
  tho_type = ['fast', 'slow'],
  dataset = ['tau', 'ljspeech', 'librispeech'],
  learning_rate = [-3, -4, -5],
  epoch = [1, 3, 10, 15, 20, 40, 70],
  # loss_type = ['l1', 'l2'],
)

###################
#transcode 1/3oct to mels using diffusion (from diffusers library)
experiment.add_plan('traindiff',
  method = ['diffusion'],
  step = ['train'],
  tho_type = ['fast', 'slow'],
  dataset = ['tau', 'ljspeech', 'librispeech', 'ljspeech_45dba'],
  learning_rate = [-3, -4, -5],
  epoch = list(range(100, -1, -5)),
  schedule = ['VE', 'VP', 'DDPM'],
  # loss_type = ['l1', 'l2'],
  diff_steps = [200, 400, 600, 800, 1000],
)

experiment.add_plan('evaldiff',
  method = ['diffusion'],
  step = ['eval', 'vocode', 'metric'],
  tho_type = ['fast', 'slow'],
  dataset = ['tau', 'ljspeech', 'librispeech', 'ljspeech_45dba'],
  evalset = ['ljspeech', 'librispeech', 'ljspeech_45dba', 'ljspeech_50dba', 'ljspeech_55dba', 'ljspeech_60dba', 'ljspeech_65dba', 'ljspeech_70dba', 'ljspeech_75dba'],
  learning_rate = [-3, -4, -5],
  epoch = list(range(0, 101, 5)),
  schedule = ['VE', 'VP', 'DDPM'],
  # loss_type = ['l1', 'l2'],
  # diff_steps = [200, 400, 600, 800, 1000],
  diff_steps = [200, 400, 600, 800, 1000],
  seed = [71, 72, 73, 74, 75],
)

###################
#transcode 1/3oct to mels using diffwave https://arxiv.org/abs/2009.09761
experiment.add_plan('traindiffwave',
  method = ['diffwave'],
  step = ['train'],
  tho_type = ['fast', 'slow'],
  dataset = ['tau', 'ljspeech', 'librispeech'],
  learning_rate = [-3, -4, -5],
  # epoch = list(range(70, -1, -5)),
  epoch = [0, 1, 3, 5, 10, 15, 20, 40, 70],
  schedule = ['VE', 'VP', 'DDPM'],
  # loss_type = ['l1', 'l2'],
  diff_steps = [0, 1, 10, 100, 500, 1000],
)

experiment.add_plan('evaldiffwave',
  method = ['diffwave'],
  step = ['eval', 'vocode', 'metric'],
  tho_type = ['fast', 'slow'],
  dataset = ['tau', 'ljspeech', 'librispeech'],
  evalset = ['ljspeech', 'librispeech'],
  learning_rate = [-3, -4, -5],
  # epoch = list(range(0, 71, 5)),
  epoch = [0, 1, 3, 5, 10, 15, 20, 40, 70],
  schedule = ['VE', 'VP', 'DDPM'],
  # loss_type = ['l1', 'l2'],
  diff_steps = [200, 400, 600, 800, 1000],
  seed = [71, 72, 73, 74, 75],
)

# experiment.add_plan('metric',
#   type = ['STOI', 'WER-w2v2', 'WER-blabla'],
#   method = ['diffusion'],
#   tho_type = ['fast', 'slow'],
#   dataset = ['tau', 'ljspeech', 'librispeech'],
#   evalset = ['ljspeech', 'librispeech'],
#   learning_rate = [-3, -4, -5],
#   epoch = list(range(0, 71, 5)),
#   schedule = ['VE', 'VP', 'DDPM'],
#   # loss_type = ['l1', 'l2'],
#   diff_steps = [1, 10, 100, 500, 1000],
# )

###############################
# ENVIRONMENTAL AUDIO METRICS
################################

# experiment.set_metric(
#   name = 'kl_pann',
#   path = 'metric',
#   output = 'kl_pann',
#   func = np.mean,
#   significance = True,
#   percent=False,
#   precision=6,
#   lower_the_better=True
#   )

# experiment.set_metric(
#   name = 'isnr',
#   path = 'metric',
#   output = 'isnr',
#   func = np.mean,
#   percent=False,
#   precision=6,
#   lower_the_better=True,
#   significance=True
#   )

# experiment.set_metric(
#   name = 'lsd',
#   path = 'metric',
#   output = 'lsd',
#   func = np.mean,
#   percent=False,
#   precision=6,
#   lower_the_better=True,
#   significance=True
#   )

# experiment.set_metric(
#   name = 'ptopafc',
#   path = 'metric',
#   output = 'ptopafc',
#   func = np.mean,
#   percent=True,
#   precision=3,
#   higher_the_better=True,
#   significance=True
#   )

###############################
# VOICE METRICS
################################

def custom_func(x):
  print('AAAAAAAA')
  print(x)
  return(np.mean(x))

# experiment.set_metric(
#   name = 'wer_w2v2',
#   path = 'metric',
#   output = 'wer_w2v2',
#   func = custom_func,
#   precision=3,
#   lower_the_better=True,
#   significance=True
#   )

# experiment.set_metric(
#   name = 'wer_whisper',
#   path = 'metric',
#   output = 'wer_whisper',
#   func = np.mean,
#   precision=3,
#   lower_the_better=True,
#   significance=True
#   )

# experiment.set_metric(
#   name = 'wer_spt',
#   path = 'metric',
#   output = 'wer_spt',
#   func = np.mean,
#   precision=3,
#   lower_the_better=True,
#   significance=True
#   )

experiment.set_metric(
  name = 'wer_w2v2_gomin',
  path = 'metric',
  output = 'wer_w2v2_gomin',
  func = np.mean,
  precision=3,
  lower_the_better=True,
  significance=True
  )

experiment.set_metric(
  name = 'wer_w2v2_grifflim',
  path = 'metric',
  output = 'wer_w2v2_grifflim',
  func = np.mean,
  precision=3,
  lower_the_better=True,
  significance=True
  )

experiment.set_metric(
  name = 'wer_whisper_gomin',
  path = 'metric',
  output = 'wer_whisper_gomin',
  func = np.mean,
  precision=3,
  lower_the_better=True,
  significance=True
  )

experiment.set_metric(
  name = 'wer_spt_gomin',
  path = 'metric',
  output = 'wer_spt_gomin',
  func = np.mean,
  precision=3,
  lower_the_better=True,
  significance=True
  )

experiment.set_metric(
  name = 'wer_ct_gomin',
  path = 'metric',
  output = 'wer_ct_gomin',
  func = np.mean,
  precision=3,
  lower_the_better=True,
  significance=True
  )

experiment.set_metric(
  name = 'wer_cr_gomin',
  path = 'metric',
  output = 'wer_cr_gomin',
  func = np.mean,
  precision=3,
  lower_the_better=True,
  significance=True
  )

# experiment.set_metric(
#   name = 'wer_w2v2_full',
#   path = 'metric',
#   output = 'wer_w2v2_full',
#   func = np.mean,
#   precision=3,
#   lower_the_better=True
# )

# experiment.set_metric(
#   name = 'wer_whisper_full',
#   path = 'metric',
#   output = 'wer_whisper_full',
#   func = np.mean,
#   precision=3,
#   lower_the_better=True
# )

# experiment.set_metric(
#   name = 'wer_spt_full',
#   path = 'metric',
#   output = 'wer_spt_full',
#   func = np.mean,
#   precision=3,
#   lower_the_better=True
# )

experiment.set_metric(
  name = 'stoi',
  path = 'metric',
  output = 'stoi',
  func = np.mean,
  precision=3,
  higher_the_better=True,
  significance=True
  )

# #This one is just a toy example for Chaymae: to be removed
# experiment.set_metric(
#   name = 'mse',
#   path = 'metric',
#   output = 'mse',
#   func = np.mean,
#   precision=3,
#   # lower_the_better=True,
#   # significance=True
#   )

# #This one is just a toy example for Chaymae: to be removed
# experiment.set_metric(
#   name = 'mse',
#   path = 'metric',
#   func = np.mean,
#   precision=3,
#   lower_the_better=True,
#   significance=True
#   )

class SettingExp:
    def __init__(self, setting, experiment, project_data_path, force_cpu=False):
        """
        Represents an experimental setting for the transcoder project. Some attributes are
        directly from doce settings, others are created in the initialisation, and depend
        on the values of the doce settings.
        
        NB: Doce setting attributes have the particularity of being "objects" types, this lead to a lot of bugs,
        especially when trying to store their values into yaml files. In this class, correct
        data types are reattributed to every doce setting attribute. 

        Args:
            setting: A doce setting object.
            experiment: The doce experiment object.
            project_data_path: Path to the project's data directory. This path must contain the data created with create_mel_tho_dataset.
            force_cpu: Whether to force CPU usage instead of GPU (default: False).
        """
        self.setting_doce = setting
        self.setting_identifier = setting.identifier()
        self.tho_type = str(setting.tho_type)
        self.step = getattr(setting, 'step', None)
        self.method = getattr(setting, 'method', None)
        #NOTE: dataset is supposed to be named "trainset" --> needs to be fixed in future versions (needs renaming every file)
        self.dataset = getattr(setting, 'dataset', None)
        self.evalset = getattr(setting, 'evalset', None)
        self.plan_name = experiment.get_current_plan().get_name()
        if self.method != "diffwave":
          self.config = f"{self.dataset}-{self.tho_type}" if self.step == 'train' else f"{self.evalset}-{self.tho_type}"
        else:
          self.config = f"{self.dataset}-{self.tho_type}-wav" if self.step == 'train' else f"{self.evalset}-{self.tho_type}-wav"
        self.project_data_path = project_data_path
        self.data_path = self.project_data_path + '/' + self.config

        #data settings
        self.setting_data = ut.load_settings(Path(project_data_path + '/' + self.config + '/' + 'settings.yaml'))
        self.metrics_type = "voice"

        if self.evalset is not None:
           if "ljspeech" in self.evalset:
              self.audio_path_gt = PROJECT_DATASET_PATH+"/LJ-SPEECH_DataSet_resampled"
           if self.evalset == "librispeech":
              self.audio_path_gt = PROJECT_DATASET_PATH+"/librispeech"
        
        ###############################
        # TRAINING AND EVAL PARAMETERS
        ################################

        #if we use "slow" third-octaves (1-s), then the batch_size and the gradient accumulation is lower
        if self.tho_type == 'slow':
            self.batch_size = 16
            self.gradient_accumulate_every = 10
        else:
            if self.method == 'diffusion':
                if self.step == 'train':
                  self.batch_size = 20
                else:
                  self.batch_size = 60
                self.gradient_accumulate_every = 10
                # self.gradient_accumulate_every = 200
            elif self.method == 'diffwave':
                if self.step == 'train':
                  self.batch_size = 20
                else:
                  self.batch_size = 10
                self.gradient_accumulate_every = 10
            else:
                self.batch_size = 2
                self.gradient_accumulate_every = 10

        #if iteration set to something else than None, stop training at a given number of iterations. Else, relies on epochs.
        self.iteration = None
        self.epoch = int(getattr(setting, 'epoch', None)) if getattr(setting, 'epoch', None) is not None else None
        self.learning_rate = 10**float(getattr(setting, 'learning_rate', None)) if getattr(setting, 'learning_rate', None) is not None else None   
        self.seed = int(getattr(setting, 'seed', None)) if getattr(setting, 'seed', None) is not None else None

        #model
        self.model_path = experiment.path.model
        self.model_name = None

        if self.step == "train":
            self.model_name = setting.identifier()+'_model'
        elif self.step in ["eval", "vocode"]:
            if self.method == 'diffusion':
                self.model_name = doce.Setting(experiment.traindiff, [setting.method, 'train', setting.tho_type, setting.dataset, setting.learning_rate, setting.epoch, setting.schedule, setting.diff_steps, setting.seed], positional=False).identifier()+'_model'
            if self.method == 'diffwave':
               self.model_name = doce.Setting(experiment.traindiffwave, [setting.method, 'train', setting.tho_type, setting.dataset, setting.learning_rate, setting.epoch, setting.schedule, setting.diff_steps, setting.seed], positional=False).identifier()+'_model'
        #model settings
        self.model_setting_name = self.model_name

        #need to change the name of the settings and of the model in case no model is found
        #and a checkpoint exists

        if self.step == "eval":
          if self.model_name is not None:
            if not os.path.exists(self.model_path+self.model_name+'.pt'):
                for k in list(range(self.epoch, 201)):
                  # temp_model_name = self.model_name
                  temp_model_setting_name = self.model_name.replace("epoch="+str(self.epoch), "epoch="+str(k))
                  temp_model_name =  temp_model_setting_name + '__chkpt_epoch' + str(self.epoch)
                  if os.path.exists(self.model_path+temp_model_name+'.pt'):
                    self.model_name = temp_model_name
                    self.model_setting_name = temp_model_setting_name

        self.model_chkpt_name = None
        self.model_chkpt_setting_name = None
        if self.step == "train":
            for k in list(range(self.epoch, 1, -1)):
                temp_model_name = self.model_name.replace("epoch="+str(self.epoch), "epoch="+str(k))
                temp_model_setting_name = temp_model_name
                if os.path.exists(self.model_path + temp_model_name + '.pt'):
                    self.model_chkpt_name = temp_model_name
                    self.model_chkpt_setting_name = temp_model_setting_name
                    break
                # temp_model_setting_name = self.model_name.replace("epoch="+str(self.epoch), "epoch="+str(j))
                # temp_model_name = temp_model_setting_name + '__chkpt_epoch' + str(k)
                # print(temp_model_name)
                # if os.path.exists(self.model_path + temp_model_name + '.pt'):
                #     self.model_chkpt_name = temp_model_name
                #     self.model_chkpt_setting_name = temp_model_setting_name
                #     break

        ###############################
        # CNN PARAMETERS (Tailleur et al., 2023)
        ################################

        self.kernel_size = int(getattr(setting, 'kernel_size', None)) if getattr(setting, 'kernel_size', None) is not None else None
        self.nb_layers = int(getattr(setting, 'nb_layers', None)) if getattr(setting, 'nb_layers', None) is not None else None
        self.dilation = int(getattr(setting, 'dilation', None)) if getattr(setting, 'dilation', None) is not None else None
        self.nb_channels = int(getattr(setting, 'nb_channels', None)) if getattr(setting, 'nb_channels', None) is not None else None

        ###############################
        # DIFFUSION PARAMETERS
        ################################

        # for old diffusion version
        # self.beta_schedule = str(getattr(setting, 'beta_schedule', None)) if getattr(setting, 'beta_schedule', None) is not None else None
        # self.loss_type = str(getattr(setting, 'loss_type', None)) if getattr(setting, 'loss_type', None) is not None else None

        self.schedule = str(getattr(setting, 'schedule', None)) if getattr(setting, 'schedule', None) is not None else None
        self.diff_steps =  int(getattr(setting, 'diff_steps', None)) if getattr(setting, 'diff_steps', None) is not None else None
        self.lr_warmup_steps =  500 if getattr(setting, 'method') in ['diffusion', 'diffwave', 'transcoder'] else None
        self.gradient_accumulation_steps =  1 if getattr(setting, 'method') in ['diffusion', 'diffwave', 'transcoder'] else None

        ###############################
        # GPU MANAGEMENT
        ################################

        useCuda = torch.cuda.is_available() and not force_cpu
        if useCuda:
            print('Using CUDA.')
            self.dtype = torch.cuda.FloatTensor
            self.ltype = torch.cuda.LongTensor
            self.device = torch.device("cuda:0")
        else:
            print('No CUDA available.')
            self.dtype = torch.FloatTensor
            self.ltype = torch.LongTensor
            self.device = torch.device("cpu")

        ###############################
        # EXPERIMENT OUTPUTS MANAGEMENT
        ################################

        #paths
        self.output_mel_path = experiment.path.mel_output
        self.output_audio_path = experiment.path.audio_output
        self.output_pann_path = experiment.path.pann_output

        #outputs of the evaluated model (audio --> 1/3oct --> mels --> audio)
        self.output_mel_name = (setting.replace('step', 'eval').identifier()+'_mel') if self.step == 'vocode' else (self.setting_identifier+'_mel')
        self.output_audio_name = (setting.replace('step', 'vocode').identifier()) if self.step == 'metric' else setting.identifier()
        self.output_pann_name = (setting.replace('step', 'vocode').identifier()) if self.step == 'metric' else setting.identifier()

        #outputs of the oracle model (audio --> mels --> audio)
        if self.step != 'train':
          self.output_audio_name_oracle = doce.Setting(experiment.baseline, ['oracle', 'vocode', setting.tho_type, setting.evalset], positional=False).identifier()
          self.output_pann_name_oracle = doce.Setting(experiment.baseline, ['oracle', 'vocode', setting.tho_type, setting.evalset], positional=False).identifier()

def step(setting, experiment):
    
    print('XXXXXXXX ONGOING SETTING XXXXXXXX')
    print(setting.identifier())
    start_time = time.time()

    setting_exp = SettingExp(setting, experiment, PROJECT_DATA_PATH)

    if setting_exp.step == "train":
        if setting_exp.method in ['transcoder', 'diffusion', 'diffwave']:
            losses, train_duration = diffusion_experiment_manager.train_dl_model(setting_exp)
            
            #saving losses in the loss folder
            for key, arr in losses.items():
                np.save(experiment.path.loss+setting.identifier()+'_'+key+'.npy', arr)

    if setting_exp.step == "eval":
        diffusion_experiment_manager.evaluate_dl_model(setting_exp)

    if setting_exp.step == "vocode":
        diffusion_experiment_manager.vocode_dl_model(setting_exp)

    if setting_exp.step == "metric":
          
        metrics, others = diffusion_experiment_manager.compute_metrics(setting_exp)
        
        for key, arr in metrics.items():
          np.save(experiment.path.metric+setting.identifier()+'_'+key+'.npy', arr)
        
        if others is not None:
          for key, arr in others.items():
            np.save(experiment.path.metric+setting.identifier()+'_'+key+'.npy', arr)

    duration = time.time() - start_time

    if setting_exp.step == "train":
      np.save(experiment.path.duration+setting.identifier()+'_train_duration.npy', train_duration)
      print("--- %s seconds ---" % (train_duration))

    np.save(experiment.path.duration+setting.identifier()+'_duration.npy', duration)
    print("--- %s seconds ---" % (duration))
        
# invoke the command line management of the doce package
if __name__ == "__main__":
  doce.cli.main(experiment = experiment, func=step)
