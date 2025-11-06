from .vieneutts import VieNeuTTS
import soundfile as sf
import os , os.path
import sys

class SpeedService:
    def __init__(
        self,
        backbone_repo="pnnbao-ump/VieNeu-TTS",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
    ):
      self.VieNeuTTSmodel = VieNeuTTS(
        backbone_repo=backbone_repo,
        backbone_device=backbone_device,
        codec_repo=codec_repo,
        codec_device=codec_device
      )
    def convertTextToSpeed(self , text):
      output_dir = "flaskr/static/output_audio"
      os.makedirs(output_dir, exist_ok=True)
      ref_audio_path = "flaskr/static/sample/id_0004.wav"
      ref_text = "flaskr/static/sample/id_0004.txt"
      ref_text = open(ref_text, "r", encoding="utf-8").read()
      print(ref_text , file=sys.stderr)
      if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return None
      ref_codes = self.VieNeuTTSmodel.encode_reference(ref_audio_path)
      i =  len([name for name in os.listdir(output_dir) if os.path.isfile(output_dir + '/' +name)]) + 1
      output_path = os.path.join(output_dir, f"output_{i}.wav")
      wav = self.VieNeuTTSmodel.infer(text , ref_codes , ref_text)
      sf.write(output_path , wav, 24000)
      output_path = '/static/output_audio/' + f"output_{i}.wav"
      return output_path
        
    