import os
import sys
import torch
from embs.model_loader import CLAPLaionModel, MSCLAPModel, PANNsModel, VGGishModel

# Get the path to the parent directory containing "beats"
current_dir = os.path.dirname(os.path.abspath(__file__))
beats_parent_dir = os.path.join(current_dir, "beats")  # Adjust as per your directory structure
sys.path.insert(0, beats_parent_dir)

# Now you can safely import BEATs
from BEATs import BEATs, BEATsConfig



class ModelsLoading:
    def __init__(self, device="cpu", emb_ckpt_dir=None):
        self.beats_model = None
        self.clap_model = None
        self.device = device
        self.emb_ckpt_dir = emb_ckpt_dir

    def load_model(self, model_name):
        model_attr = f"{model_name}_model"
        if getattr(self, model_attr) is None:
            load_method = getattr(self, f"load_{model_name}")
            return load_method()
        else:
            return getattr(self, model_attr)
        
    def load_beats(self):
        # Load beats model
        beats_ckpt_relative_path = "./beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
        beats_ckpt_full_path = os.path.abspath(beats_ckpt_relative_path)
        checkpoint = torch.load(beats_ckpt_full_path)

        cfg = BEATsConfig(checkpoint['cfg'])
        beats_model = BEATs(cfg)
        beats_model.load_state_dict(checkpoint['model'])
        beats_model.eval()
        beats_model.to(self.device)
        return beats_model

    def load_clap(self):
        self.clap_model = CLAPLaionModel(type="audio", device=self.device, emb_ckpt_dir=self.emb_ckpt_dir)
        self.clap_model.load_model()
        return self.clap_model

    def load_pann(self):
        self.pann_model = PANNsModel(variant="wavegram-logmel", device=self.device, emb_ckpt_dir=self.emb_ckpt_dir)
        self.pann_model.load_model()
        return self.pann_model
    
    def load_msclap(self):
        self.msclap_model = MSCLAPModel(type="2023", device=self.device, emb_ckpt_dir=self.emb_ckpt_dir)
        self.msclap_model.load_model()
        return self.msclap_model
    
    def load_vggish(self):
        self.vggish_model = VGGishModel(device=self.device, emb_ckpt_dir=self.emb_ckpt_dir)
        self.vggish_model.load_model()
        return self.vggish_model
