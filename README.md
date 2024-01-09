# Chest X-ray Report Generation 

## Goal
![image](https://github.com/hanajibsa/CXR_report_generation/assets/115071643/470cc7b1-f5a7-46d8-b798-d83de3fd8445)

Image captioning 기술을 바탕으로 chest x-ray 이미지를 읽고 레포트를 생성하는 모델을 개발해 의료 서비스의 자동화에 기여하고자 하였습니다.


먼저 Chest X-ray란 폐와 심장, 갈비뼈, 척추 등 흉부 내부의 구조를 보여주는 영상으로, 다양한 폐 질환, 심장 질환, 가슴 부상 등을 진단하는 데 사용합니다. 이런 의료 영상은 의료 전문가가 판독하고 해석하며, 검사 결과는 의료 판독문(즉, medical reports)을 통해 전달됩니다. 일반적으로 의사는 하나의 레포트를 작성하는데 5~10분 소요되며, 하루에 100개 정도의 레포트을 생성하기 때문에 시간이 많이 소요됩니다. 이러한 상황에서 Image captioning 기술을 통해 레포트 작성을 자동화하면, 판독 시간을 단축시키고 비용 절감을 가능하게 함으로써 의료 시스템의 전반적인 효율성을 개선하고, 더 많은 환자들에게 고품질의 의료 서비스를 제공하는 데 기여할 수 있을 것이라 생각했습니다.

## Dataset 
<img src="https://github.com/hanajibsa/CXR_report_generation/assets/115071643/9f1f8058-1614-494c-b0ab-c9233f7e27be.png" width="350" height="350"/>
<img src="https://github.com/hanajibsa/CXR_report_generation/assets/115071643/fd53a6cd-63fb-4114-93f6-1dec2390d6d1.png" width="400" height="350"/>

MIMIC-CXR 데이터베이스는 dicom 형식의 흉부 엑스레이 이미지와 텍스트로 된 방사선학 레포트으로 이루어진 대규모 공개 데이터셋 입니다. 이 데이터셋은 약  65,379명의 환자로 구성되어 있고 각 환자 당  3-4개의 연구를 수행하였습니다. 각 연구 당 하나의 레포트와 약 2개의 엑스레이 이미지를 포함합니다. 
제한된 컴퓨팅 리소스를 고려할 때, 특정 질환에 초점을 맞추는 것이 데이터 처리의 효율을 높이는 방법이라 생각했습니다. 그래서 특히 전세계적으로 가장 높은 유병률을 가지는 폐 질환에 특화된 레포트를 생성하는 모델을 만들어서 정확성과 효율성을 높이고자 하였습니다.

가장 일반적으로 사용이 되는 PA 이미지 만을 사용해서 일관성을 유지하였고 폐 관련 라벨들만을 포함하는 텍스트를 사용하여 폐 질환에 특화된 모델을 만들었습니다.
- Image pre-processing: 실제 데이터는 이미지에 텍스트가 포함되어 있는 경우가 많습니다. 이 텍스트는 학습에 부정적인 영향을 미칠 수 있으므로, 이미지 상단의 왼쪽과 오른쪽을 masking하고 이미지의 하단을 잘라 크기를 (224,224)로 조정하여 폐에 집중한 이미지를 만들었습니다. 
- Text pre-processing: 폐에 특화된 레포트를 생성하기 위해 FINDINGS와 IMPRESSION에서 폐에 관련된 문장을 하나씩 추출하여 json 파일을 만들어 사용하였습니다. 

## Methodology
![image](https://github.com/hanajibsa/CXR_report_generation/assets/115071643/f85c9494-4b0a-4ea3-a839-f014229757da)

Vision-language understanding task와  generation task를 통합한 VLP 프레임워크인 BLIP 파이프라인을 사용하였습니다. 각각 이미지 인코더와 텍스트 인코더, 디코더로 다음과 같은 모델을 사용하였습니다. 
- Image encoder: ViT
- Text encoder/decoder: medical domain-specific BERT models
    - BlueBERT
    - PubMedBERT
    - BioBERT

## Result 
<img src="https://github.com/hanajibsa/CXR_report_generation/assets/115071643/f48ad037-d314-48ff-a76f-d57d61d9400a.png" width="500" height="100"/>

앞서 설명드린 세 개의 biomedical BERT를 사용하여 최종 모델의 성능을 비교한 표입니다. 저희는 이 모델들을 네 가지의 텍스트 유사성 지표를 사용해 평가했습니다. 이 표에서 score는 높을수록 생성된 문장이 실제 문장과 유사함을 의미하는데, BlueBERT가 가장 좋은 성능을 나타내는 것을 볼 수 있습니다.

따라서 저희는 BlueBERT를 이용해 chest x-ray 이미지를 입력하면 findings과 impression를 각각 한 문장씩 생성하여 레포트를 만드는 모델을 개발하였습니다.

## Code 
```
pip install -r requirements.txt
```

BlueBERT
```
python train.py --checkpoint CXR-Report-Generation/output/Pretrain/findigns/checkpoint_80.pth --pretrain_BERT bert-large-uncased
```

PubMedBERT
```
python train.py --pretrain_BERT microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
```

BioBERT
```
python train.py --pretrain_BERT pritamdeka/BioBert-PubMed200kRCT
```
