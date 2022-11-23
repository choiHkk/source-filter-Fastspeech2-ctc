## Introduction
1. FastSpeech2 오픈 소스를 활용해 FastPitchFormant를 구현하고 한국어 데이터셋(KSS)을 사용해 빠르게 학습합니다.
2. 논문에서 제시한 방법은 Phoneme-Level Duration과 Pitch를 사용하지만, 본 레포지토리에서는 Frame-Level을 기준으로 사용합니다. 
3. 기존 오픈소스는 MFA기반 preprocessing을 진행한 상태에서 학습을 진행하지만 본 레포지토리에서는 alignment learning 기반 학습을 진행하고 preprocessing으로 인해 발생할 수 있는 디스크 용량 문제를 방지하기 위해 data_utils.py로부터 학습 데이터가 feeding됩니다.
4. conda 환경으로 진행해도 무방하지만 본 레포지토리에서는 docker 환경만 제공합니다. 기본적으로 ubuntu에 docker, nvidia-docker가 설치되었다고 가정합니다.
5. GPU, CUDA 종류에 따라 Dockerfile 상단 torch image 수정이 필요할 수도 있습니다.
6. preprocessing 단계에서는 학습에 필요한 transcript와 stats 정도만 추출하는 과정만 포함되어 있습니다.
7. 그 외의 다른 preprocessing 과정은 필요하지 않습니다.

## Dataset
1. download dataset - https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
2. `unzip /path/to/the/kss.zip -d /path/to/the/kss`
3. `mkdir /path/to/the/source-filter-FastSpeech2-ctc/data/dataset`
4. `mv /path/to/the/kss.zip /path/to/the/source-filter-FastSpeech2-ctc/data/dataset`

## Docker build
1. `cd /path/to/the/source-filter-FastSpeech2-ctc`
2. `docker build --tag source-filter-FastSpeech2-ctc:latest .`

## Training
1. `nvidia-docker run -it --name 'source-filter-FastSpeech2-ctc' -v /path/to/source-filter-FastSpeech2-ctc:/home/work/source-filter-FastSpeech2-ctc --ipc=host --privileged source-filter-FastSpeech2-ctc:latest`
2. `cd /home/work/source-filter-FastSpeech2-ctc`
3. `cd /home/work/source-filter-FastSpeech2-ctc/hifigan`
4. `unzip generator_universal.pth.tar.zip .`
5. `cd /home/work/source-filter-FastSpeech2-ctc`
6. `ln -s /home/work/source-filter-FastSpeech2-ctc/data/dataset/kss`
7. `python preprocess.py ./config/kss/preprocess.yaml`
8. `python train.py -p ./config/kss/preprocess.yaml -m ./config/kss/model.yaml -t ./config/kss/train.yaml`
9. arguments
  * -p : preprocess config path
  * -m : model config path
  * -t : train config path
10. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Tensorboard losses
![source-filter-FastSpeech2-ctc-tensorboard-losses](https://user-images.githubusercontent.com/69423543/203502687-4c770ff8-ddc0-4432-bb41-fa35f4fd43ab.png)

## Tensorboard Stats
![source-filter-FastSpeech2-ctc-tensorboard-stats](https://user-images.githubusercontent.com/69423543/203502730-ecc3d977-20d7-4f2e-a3d8-b839a859f5ec.png)
![source-filter-FastSpeech2-ctc-tensorboard-alignments](https://user-images.githubusercontent.com/69423543/203502745-a24635cb-0721-45c9-b038-bd3936fbea59.png)


## Reference
1. [FastPitchFormant: Source-filter based Decomposed Modeling for Speech Synthesis](https://arxiv.org/abs/2106.15123)
2. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
3. [One TTS Alignment To Rule Them All](https://arxiv.org/pdf/2108.10447.pdf)
4. [FastSpeech2 github](https://github.com/ming024/FastSpeech2)
5. [FastPitchFormant](https://github.com/keonlee9420/FastPitchFormant)
