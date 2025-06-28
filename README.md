
#  whisper-mini-ta: Fine-tuned Whisper Model for Tamil ASR

📍 Model on Hugging Face: [ragunath-ravi/whisper-mini-ta](https://huggingface.co/ragunath-ravi/whisper-mini-ta)

**`whisper-mini-ta`** is a fine-tuned version of OpenAI's Whisper model, adapted specifically for **Tamil automatic speech recognition (ASR)**. It is trained on the [TamilVoiceCorpus](https://huggingface.co/datasets/ragunath-ravi/TamilVoiceCorpus), an open-source speech dataset collected from public sources.

---
# Whisper Mini ta - RAGUNATH RAVI

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the whisperaudio dataset. It achieves the following results on the evaluation set:
- Loss: 0.0905
- WER: 18.7042

## Model description

This is a Whisper Small model fine-tuned specifically for Tamil language automatic speech recognition (ASR). The model has been adapted to recognize and transcribe Tamil speech with higher accuracy than the base OpenAI Whisper model.

The fine-tuning process focused on optimizing the model for Tamil phonetics, vocabulary, and speech patterns using a custom web-scraped dataset. The model uses the Whisper architecture, which employs a Transformer-based encoder-decoder architecture with attention mechanisms specifically designed for speech recognition tasks.

This model is particularly well-suited for Tamil speech recognition applications, achieving a Word Error Rate (WER) of 18.70% on the test set, demonstrating significant improvements over the base model for Tamil language speech.

## Intended uses & limitations

### Intended uses

- Transcription of Tamil speech in audio and video content
- Voice command systems for Tamil speakers
- Accessibility tools for Tamil-speaking users
- Documentation of Tamil audio content
- Subtitling and captioning services for Tamil media

### Limitations

- The model may struggle with heavily accented Tamil speech or regional dialects that were not well-represented in the training data
- Performance may degrade with noisy audio inputs or low-quality recordings
- The model might have difficulty with specialized terminology or domain-specific language not present in the training data
- The model is specifically trained for Tamil and will not perform well on other languages

## Training and evaluation data

The model was fine-tuned on a custom web-scraped dataset called "whisperaudio" (available at ragunath123/whisperaudio on Hugging Face). This dataset consists of Tamil speech audio paired with accurate transcriptions.

For training, 12,000 samples were used from the dataset, while 3,000 samples were used for evaluation. The audio was processed by resampling from 48kHz to 16kHz to match Whisper's requirements.

The dataset includes a diverse range of Tamil speech samples, which helps the model generalize across different speakers, accents, and content types.

## Training procedure

### Preprocessing

- Audio files were resampled from their original sampling rate to 16kHz
- Log-Mel spectrograms were extracted as input features using the Whisper feature extractor
- Text was tokenized using the Whisper tokenizer configured specifically for the Tamil language
- Special care was taken to handle the tokenization of Tamil characters correctly

### Framework versions

- Transformers 4.40.2
- PyTorch 2.7.0+cu126
- Datasets 3.5.1
- Tokenizers 0.19.1

### Training hyperparameters

The following hyperparameters were used during training:
- Learning rate: 1e-05
- Train batch size: 32
- Evaluation batch size: 16
- Seed: 42
- Optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- Learning rate scheduler: Linear with warmup
- Learning rate scheduler warmup steps: 500
- Total training steps: 4000
- Mixed precision training: Native AMP
- Gradient accumulation steps: 1
- Evaluation strategy: Steps (every 1000 steps)
- Gradient checkpointing: Enabled
- FP16 training: Enabled

### Training results

| Training Loss | Epoch   | Step | Validation Loss | WER     |
|:-------------:|:-------:|:----:|:---------------:|:-------:|
| 0.0585        | 2.6667  | 1000 | 0.0872          | 20.2050 |
| 0.0123        | 5.3333  | 2000 | 0.0905          | 18.7042 |
| 0.0047        | 8.0     | 3000 | 0.1033          | 18.7719 |
| 0.0015        | 10.6667 | 4000 | 0.1116          | 18.8828 |

The model achieved its best performance at epoch 5.3 (step 2000) with a WER of 18.7042%.

### Model configuration

The model was configured specifically for Tamil language transcription:
- Language set to "tamil"
- Task set to "transcribe"
- Forced decoder IDs were set to None to allow the model more flexibility in generation

## Evaluation

The model was evaluated using the Word Error Rate (WER) metric, which measures the percentage of words incorrectly transcribed. The final model achieved a WER of 18.70%, indicating that approximately 81.3% of words were correctly transcribed.

The evaluation was performed on a held-out test set of 3,000 samples from the whisperaudio dataset, ensuring a fair assessment of the model's performance on unseen data.

## Usage

```python
from transformers import pipeline
import torch

# Load the model
asr = pipeline(
    "automatic-speech-recognition",
    model="ragunath-ravi/whisper-mini-ta",
    device=0 if torch.cuda.is_available() else "cpu"
)

# Transcribe audio
result = asr("path_to_audio_file.wav", language="ta", task="transcribe")
print(result["text"])
