> Speech Denoising
## Abstract
Mean Teacher는 Semi-Supervised Learnimg으로 Labeling과 UnLabeling 되어 있는 Image들을 
classification 하는데 사용

Mean Teachers에서는 Hard Training, 같은 Input에 일부러 서로 다른 Noise 입히고 일관된 Prediction이 나오도록 하는 Consistency Loss 사용

ASR Denoising은 Noisy가 들어간 Speech Data에서 Noisy가 들어가지 않은 Speech Data와 일관된 결과가 나오도록
학습한다는 점이 Mean Teacher와 비슷하다고 생각하여 적용

## Implement

**Training**

`python main.py --epoch 100 --batch-size (GPU) --warm-steps 750000(batch 8 기준)
--input-dim 161 --max-len 1000 --gpu (GPU) --consistency 100.0 --consistency-rampup 3`

## Model
- Clean Data의 Logit은 Target(text)와 cross_entropy (Classification Loss)

- Noisy Data의 Logit은 Clean Data Logit과 MSE LOSS (Consistency Loss)

- EMA_Model Weight(Input: Noisy Data) 는 **Training** 하지않고 Student Weight(Clean data)에서 **Exponential Moving Average**로 Update

![1](https://user-images.githubusercontent.com/76771847/122905728-6e0a8f00-d38c-11eb-9907-b0449397225e.png)

![c](https://user-images.githubusercontent.com/76771847/126824793-0e5bcfa1-86a5-4705-a08f-426aaffa8961.png)

## Reference

Mean_Teachers: https://arxiv.org/pdf/1703.01780

Github: https://github.com/park-cheol/Pytorch-Mean_Teachers

ASR-Transformer: https://arxiv.org/pdf/1706.02737

Github: https://github.com/park-cheol/ASR-Transformer

## Dataset

clovacall: https://github.com/clovaai/ClovaCall

Noisy_Data: ClovaCall Dataset에다가 흔히 발생되는 Noise wav 파일들을 SNR 비율(0)로 입힘 

`cd NoiseInjection`

`python injection.py --ARGUMENT..`


