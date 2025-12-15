import torch
import os
import pann
import pandas as pd

class PANNsModel():
    """
    Kong, Qiuqiang, et al., "Panns: Large-scale pretrained audio neural networks for audio pattern recognition.",
    IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

    Specify the model to use (cnn14-32k, cnn14-16k, wavegram-logmel).
    You can also specify wether to send the full provided audio or 1-s chunks of audio (cnn14-32k-1s). This was shown 
    to have a very low impact on performances.
    """
    def __init__(self):
                # Load labels from the first column of the Excel file using pandas
        df = pd.read_excel('./utils/audioset_tvb.xlsx', usecols=[0])
        self.labels_str = df.iloc[:, 0].dropna().tolist()

    def load_model(self, type="cnn14", ckpt_dir="./pann/ckpt"):
        self.type = type
        # Check if CUDA is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")

        if type == "wavegram-logmel":
            dl_link = "https://zenodo.org/records/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth"
        if type == "cnn14":
            dl_link = "https://zenodo.org/record/3576403/files/Cnn14_mAP%3D0.431.pth"

        dl_dir = os.path.join(ckpt_dir,"https://zenodo.org/records/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth")
        if not os.path.exists(dl_dir):
            print("Download pretrained checkpoints of Cnn14.")
            os.makedirs(dl_dir, exist_ok=True)
            os.system(
                f"wget -P {ckpt_dir} %s"
                % (dl_link)
            )

        if self.type == "wavegram-logmel":
            self.model = pann.Wavegram_Logmel_Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            state_dict = torch.load(f"{ckpt_dir}/Wavegram_Logmel_Cnn14_mAP=0.439.pth", map_location=self.device)
            self.model.load_state_dict(state_dict["model"])
        if self.type == "cnn14":
            features_list = ["2048", "logits"]
            self.model = pann.Cnn14(
                features_list=features_list,
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            state_dict = torch.load(f"{ckpt_dir}/Cnn14_mAP=0.431.pth", map_location=self.device)
            self.model.load_state_dict(state_dict["model"])
        if self.type == "cnn14declev":
            self.model = pann.Cnn14_DecisionLevelMax(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            state_dict = torch.load(f"{ckpt_dir}/Cnn14_DecisionLevelMax_mAP=0.385.pth", map_location=self.device)
            self.model.load_state_dict(state_dict["model"])

        self.model.eval()
        self.model.to(self.device)

        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'PANNs: {self.num_params}')

    def get_embedding(self, audio):
        audio = torch.from_numpy(audio).float().to(self.device)
        with torch.no_grad():
            outputs = self.model.forward(audio)
            logits = outputs["clipwise_output"]
            if 'cnn14' in self.type:
                emb = outputs["2048"]
            else:
                emb = outputs["embedding"]
        return emb, logits
    
    def get_logits(self, audio):
        audio = torch.from_numpy(audio).float().to(self.device)
        with torch.no_grad():
            outputs = self.model.forward(audio)
            if self.type == 'cnn14declev':
                logits = outputs["framewise_output"]
            else:
                logits = outputs["clipwise_output"]
        return logits
    