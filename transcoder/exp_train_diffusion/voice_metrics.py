import jiwer
import torch
import torchaudio
import os
import numpy as np
from pystoi import stoi
import scipy
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from speechbrain.inference.interfaces import Pretrained

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

class WerMetricW2V2():
    def __init__(self, device=torch.device("cpu")):
        self.device = device

        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

        self.model = self.bundle.get_model().to(device)

        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        self.decoder = GreedyCTCDecoder(labels=self.bundle.get_labels())

    def calculate(self, audio_test, audio_target, sr):
        transcript_test = self.get_transcript(audio_test, sr)
        transcript_target = self.get_transcript(audio_target, sr)

        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        truth_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )

        return wer
    
    def get_transcript(self, audio, sr):
        audio_input = audio.to(self.device)
        if sr != self.bundle.sample_rate:
            audio_input = torchaudio.functional.resample(audio, sr, self.bundle.sample_rate).to(self.device)

        with torch.inference_mode():
            emission, _ = self.model(audio_input)

        transcript = self.decoder(emission[0])
        transcript = transcript.replace("|", " ")

        return transcript
    
    def get_wer(self, transcript_target, transcript_test):
        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        truth_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )
        return(wer)

class WerMetricWhisper():
    def __init__(self, device=torch.device("cpu")):
        self.device = device

        # Load the Whisper model and processor
        # model_name = "openai/whisper-large-v3"
        # model_name = "openai/whisper-large-v2"
        # model_name = "openai/whisper-large"
        # model_name = "openai/whisper-medium"
        # model_name = "openai/whisper-small"
        # model_name = "openai/whisper-base"
        model_name = "openai/whisper-tiny"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)

        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        self.sr = 16000

    def calculate(self, audio_test, audio_target, sr):
        transcript_test = self.get_transcript(audio_test, sr)
        transcript_target = self.get_transcript(audio_target, sr)

        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        truth_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )

        return wer
    
    def get_transcript(self, audio, sr):
        audio_input = audio
        if sr != self.sr:
            audio_input = torchaudio.functional.resample(audio, sr, self.sr)

        # Ensure waveform is 1D (mono) for the model if it is 2D (stereo)
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(dim=0, keepdim=True)
        
        audio_input = audio_input.to(torch.device("cpu"))

        inputs = self.processor(audio_input.squeeze(), sampling_rate=self.sr, return_tensors="pt", device=self.device).to(self.device)

        # Generate transcription
        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        # Decode the transcription
        transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        return transcription[0]
    
    def get_wer(self, transcript_target, transcript_test):
        try:
            wer = jiwer.wer(
                transcript_target,
                transcript_test,
                truth_transform=self.transforms,
                hypothesis_transform=self.transforms,
            )
        except ValueError:
            print('ValueError: one of the transcript might be empty')
            wer = 1.0
        return(wer)

class WerMetricSPT():
    def __init__(self, device=torch.device("cpu")):
        # self.device = torch.device("cpu")
        self.device = device
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr").to(self.device)

        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        self.sr = 16000

    def calculate(self, audio_test, audio_target, sr):
        transcript_test = self.get_transcript(audio_test, sr)
        transcript_target = self.get_transcript(audio_target, sr)

        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        truth_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )

        return wer
    
    def get_transcript(self, audio, sr):
        audio_input = audio
        if sr != self.sr:
            audio_input = torchaudio.functional.resample(audio, sr, self.sr)

        # Ensure waveform is 1D (mono) for the model if it is 2D (stereo)
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(dim=0, keepdim=True)
        
        # audio_input = audio_input.to(torch.device("cpu"))

        # inputs = self.processor(audio_input.squeeze(), sampling_rate=self.sr, return_tensors="pt", device=self.device).to(self.device)
        inputs = self.processor(audio_input.detach().cpu().squeeze(0).numpy(), sampling_rate=self.sr, return_tensors="pt")

        input_features = inputs["input_features"]
        attention_mask = inputs["attention_mask"]

        # Generate the transcription
        generated_ids = self.model.generate(input_features.to(self.device), attention_mask=attention_mask.to(self.device))

        # Decode the generated ids to get the transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        return transcription[0]
    
    def get_wer(self, transcript_target, transcript_test):
        try:
            wer = jiwer.wer(
                transcript_target,
                transcript_test,
                truth_transform=self.transforms,
                hypothesis_transform=self.transforms,
            )
        except ValueError:
            print('ValueError: one of the transcript might be empty')
            wer = 1.0
        return(wer)

class WerMetricSB():
    def __init__(self, device=torch.device("cpu"), asr_short_name="ct"):
        # self.device = torch.device("cpu")
        self.device = device
        # asr_name = "asr-wav2vec2-commonvoice-14-en"
        # asr_name = "asr-streaming-conformer-librispeech"
        # asr_name = "asr-conformersmall-transformerlm-librispeech"
        # asr_name = "asr-crdnn-commonvoice-14-en"
        # asr_name = "asr-crdnn-rnnlm-librispeech"
        if asr_short_name == "cst":
            asr_name = "asr-conformersmall-transformerlm-librispeech"
        if asr_short_name == "sc":
            asr_name = "asr-streaming-conformer-librispeech"
        if asr_short_name == "ct":
            asr_name = "asr-conformer-transformerlm-librispeech"
        if asr_short_name == "cr":
            asr_name = "asr-crdnn-rnnlm-librispeech"
            
        self.asr_model = SpeechbrainEncoderDecoderASR.from_hparams(source=f"speechbrain/{asr_name}", savedir=f"pretrained_models/{asr_name}", run_opts={"device":device})

        self.transforms = jiwer.Compose(
            [
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        self.sr = 16000

    def calculate(self, audio_test, audio_target, sr):
        transcript_test = self.get_transcript(audio_test, sr)
        transcript_target = self.get_transcript(audio_target, sr)

        wer = jiwer.wer(
                        transcript_target,
                        transcript_test,
                        truth_transform=self.transforms,
                        hypothesis_transform=self.transforms,
                    )

        return wer
    
    def get_transcript(self, audio, sr):
        audio_input = audio
        if sr != self.sr:
            audio_input = torchaudio.functional.resample(audio, sr, self.sr)

        # Ensure waveform is 1D (mono) for the model if it is 2D (stereo)
        if audio_input.ndim > 1:
            audio_input = audio_input.mean(dim=0, keepdim=True)
        
        # audio_input = audio_input.to(torch.device("cpu"))
        transcription = self.asr_model.transcribe_file(audio_input)
        
        return transcription
    
    def get_wer(self, transcript_target, transcript_test):
        try:
            wer = jiwer.wer(
                transcript_target,
                transcript_test,
                truth_transform=self.transforms,
                hypothesis_transform=self.transforms,
            )
        except ValueError:
            print('ValueError: one of the transcript might be empty')
            wer = 1.0
        return(wer)

class SpeechbrainEncoderDecoderASR(Pretrained):
    """A ready-to-use Encoder-Decoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech. The given YAML must contain the fields
    specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.inference.ASR import EncoderDecoderASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = EncoderDecoderASR.from_hparams(
    ...     source="speechbrain/asr-crdnn-rnnlm-librispeech",
    ...     savedir=tmpdir,
    ... )  # doctest: +SKIP
    >>> asr_model.transcribe_file("tests/samples/single-mic/example2.flac")  # doctest: +SKIP
    "MY FATHER HAS REVEALED THE CULPRIT'S NAME"
    """

    HPARAMS_NEEDED = ["tokenizer"]
    MODULES_NEEDED = ["encoder", "decoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer
        self.transducer_beam_search = False
        self.transformer_beam_search = False
        if hasattr(self.hparams, "transducer_beam_search"):
            self.transducer_beam_search = self.hparams.transducer_beam_search
        if hasattr(self.hparams, "transformer_beam_search"):
            self.transformer_beam_search = self.hparams.transformer_beam_search

    def transcribe_file(self, waveform, **kwargs):
        """Transcribes the given audiofile into a sequence of words.

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        # waveform = self.load_audio(path, **kwargs)
        # Fake a batch:
        # audio = torch.from_numpy(waveform)
        # batch = audio.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.transcribe_batch(
            waveform, rel_length
        )
        return predicted_words[0]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        if self.transformer_beam_search:
            encoder_out = self.mods.transformer.encode(encoder_out, wav_lens)
        return encoder_out

    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        list
            Each waveform in the batch transcribed.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            if self.transducer_beam_search:
                inputs = [encoder_out]
            else:
                inputs = [encoder_out, wav_lens]
            predicted_tokens, _, _, _ = self.mods.decoder(*inputs)
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full transcription - note: no gradients through decoding"""
        return self.transcribe_batch(wavs, wav_lens)

class STOIMetric():
    def __init__(self, device=torch.device("cpu")):
        self.device = device
        self.sr = 16000

    def calculate(self, audio_test, audio_target, sr):

        d = stoi(audio_target[0], audio_test[0], sr, extended=False)
        
        return d