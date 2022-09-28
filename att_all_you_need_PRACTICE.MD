# att_all_you_need.py 코드 결과 설명
[yonlu 모듈](https://github.com/MinSong2/yonlu)에서 Transformer를 구현해 놓은 att_all_you_need.py 코드를 Transformer의 구조에 맞춰서 설명하고자 합니다. 

# 실행 코드
**main_code.img**

# 코드 해석
## 1. Transformer 모델 생성
```python
make_model(src_vocab=V, tgt_vocab=V, N=2)
```
* 메인코드 첫째줄에서 `V=11`로 선언하여 임베딩을 할 단어들의 개수 즉, 단어 집합의 크기를 11로 만들어 줍니다.(V=num_embeddings) <br>
따라서, 첫번째 인자 `src_vocab=V`에서 encoder에 입력되는 단어 집합의 크기를 11로 만들고, 두번째 인자 `tgt_vocab=V`에서 decoder에 입력되는 단어 집합의 크기도 11로 만듭니다. <br>
Transformer 논문에서는 Encoder와 Decoder를 각각 6개 layer로 쌓았는데, 본 실행 코드에서는 `N=2`로 입력하여 2개의 layer로 모델을 구축합니다. 
다른 하이퍼파라미터는 모델에서 제시한 값을 그대로 default로 사용합니다. d_model=512, d_ff=2048, h=8, dropout=0.1 (이 때, d_model=embedding_dim 즉, 임베딩 벡터의 차원) <br>
**make_module.img**

* `EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)`를 이용해서 아래 그림과 같은 전체 transformer 틀을 구축합니다. <br>
**transformer_structure.img** <br>
가장 중요한 `encoder`와 `decoder`를 구성하고, encoder에 입력될 임베딩 벡터(`src_embed`)와 decoder에 입력될 임베딩 벡터(`tgt_embed`)를 생성합니다. 
마지막으로 decoder의 출력값에 Linear와 Softmax를 수행하도록 하는 `generator`를 구성하여 전체 틀을 구축합니다. <br>
**encoder_decoder.img** <br>

다음으로는 Transformer를 구성하는 각 부분을 살펴보겠습니다.  

**1) 인코더 : `Encoder(layer, N)`**
**encoder_structure.img** <br>
- `N=2`로 지정하였으므로 encoder layer를 2개 생성합니다. <br>
  그리고 `LayerNorm()`을 적용하여, encoder layer를 구성하는 2개의 sub-layer(Multi-Head Attention과 Feed Forward)에 Add & Layer-Normalization을 수행합니다. <br>
**encoder.img** <br>
**layer-norm.img** <br>
**layer_norm_structure.img** <br>
- 각각의 Encoder layer는 2개의 sub-layer(Multi-Head Attention과 Feed Forward)로 구성되는데, `MultiHeadedAttention` 클래스를 이용해서 self-attention 층을 만들고, 
`PositionwiseFeedForward`를 사용하여 Feed Forward 층을 만듭니다. 
이렇게 만든 sub-layer를 `EncoderLayer(size, self_attn, feed_forward, dropout)`의 두번째와 세번째 인자에 넣어 하나의 Encoder layer를 생성하게 됩니다. <br>
**EncoderLayer.img**

###### `MultiHeadedAttention`
**multihead_structure.img** <br>
+ Q, K, V 벡터를 h번의 Linear projection을 통해 `d_k=d_model//h` 차원으로 쪼갠 h개의 Q, K, V 벡터를 생성 → 각 Q, K, V에 대해서 Scaled Dot-Product Attention(`attention()`) 수행 →  h개의 출력값을 하나로 concat → 최종 Linear 결합을 통해 다시 `d_model` 크기로 출력 
+ 이렇게 multi-head로 구성할 경우 병렬 처리를 통해 여러 representation subspace를 제공하고 head별로 다양한 특성에 집중할 수 있도록 해 attention layer의 성능을 향상시킵니다.  
**multihead.img** <br>
**attention.img**

###### `PositionwiseFeedForward`
**FF_structure.img** 
+ Fully-connected feed-forward network는 Linear transformation으로 
$\text{FFN}(x)=\text{max}(0, x W_1 + b_1)W_2 + b_2$ 를 적용하며, position마다 동일한 구조를 적용하지만 Layer마다 파라미터는 각각 다르게 사용합니다. <br>
**positionwise_feed_forward.img**

**2) 디코더 : `Decoder(layer, N)`**
**Decoder_structure.img** <br>
- 디코더도 인코더와 마찬가지로 `N=2`로 지정하였으므로 decoder layer를 2개 생성하고, 각 sub-layer에 `LayerNorm()`을 적용하여 Add & Layer-Normalization을 수행합니다. <br>
**Decoder.img** <br>
- 단, 디코더에서 각각의 Decoder layer는 Multi-Head Attention과 Feed Forward sub-layer 이전에 Masked Multi-Head Attention이라는 sub-layer가 추가되어 
총 3개의 sub-layer를 가집니다. 디코더에서 Masked Multi-Head Attention은 <U>self-attention</U>으로 `MultiHeadedAttention`에 
<U>디코더의 임베딩 벡터</U>(출력 시퀀스 이전 위치에만 attention할 수 있도록 미래 위치를 마스킹한 벡터)를 입력하여 만들고, 
디코더의 Multi-Head Attention은 <U>Encoder-Decoder attention</U>으로 `MultiHeadedAttention`에 <U>인코더의 임베딩 벡터</U>를 입력하여 만듭니다. 
- 이렇게 만든 sub-layer를 `DecoderLayer(size, self_attn, src_attn, feed_forward, dropout)`의 두번째와 세번째 인자에 넣어 하나의 Decoder layer를 생성하게 됩니다.  
**Decoder-Layer.img** 

**3) 임베딩 벡터 : `nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding())`**
**embedding_structure.img** <br>
- 인코더의 입력되는 임베딩 벡터의 경우 `src_vocab`에 대해서 단어 임베딩과 위치임베딩을 순차적으로 수행(`nn.Sequential()`)하고, 디코더에 입력되는 임베딩 벡터의 경우 
`tgt_vocab`에 대해서 순차적으로 수행합니다. 
- `Embeddings(d_model, src_vocab)`에서는 각 단어를 `d_model` 크기로 임베딩하고(src_vocab x d_model 크기 행렬 생성),
 `PositionalEncoding(d_model, dropout)`을 통해 위치 임베딩까지 sequential하게 순차적으로 수행합니다. 

**4) 디코더 출력값 Linear & Softmax : Generator(d_model, vocab)**
**generator_structure.img** <br>
- 마지막으로 디코더에서 출력된 실수(float) 벡터를 단어로 바꾸기 위해 우선 `nn.Linear(d_model, vocab)`으로 벡터 크기를 `d_model`에서 
총 단어 집합의 크기인 `vocab`으로 바꿔줍니다.
- 그리고 Softmax 층에서 해당 점수를 확률로 변환하여 가장 높은 확률값에 해당하는 단어를 선택하고, 이 단어가 해당 time step의 출력으로 반환됩니다.  <br>
**generator.img**

## 2. `model_opt = NoamOpt(model_size, factor, warmup, optimizer)`
- optimizer는 논문에서 사용한 그대로 `Adam optimizer`, `beta1 = 0.9`, `beta2= 0.98`, `eps = 1e-9`를 사용합니다. 
- (PyTorch에서의 optimizer와 loss calculation 관련해서 조금 더 공부 필요) <br>
**NoamOpt.img**

## 3. `run_epoch(data_iter, model, loss_compute)`
- 모델이 제대로 작동하는지 확인하기 위해 `data_gen(V, batch, nbatches)`을 통해 직접 데이터 및 batch를 생성하여 첫번째 인자로 입력합니다. 
- `model.train()`에서 1 epoch 당 [1, V=11) 사이 랜덤 숫자로 (`batch`=30, 10) 사이즈의 미니배치 20개(`nbatches`)씩 생성하여 학습을 시키고, 
`model.eval()`에서는 5개(`nbatches`)씩 생성하여 평가를 진행합니다.
- 두번째 인자에는 앞서 `make_model()`을 통해 만든 모델을 입력합니다.
- 세번째 인자인 loss_compute에는 `SimpleLossCompute(model.generator, criterion, model_opt)`를 수행합니다. 
`model.generator`로 얻은 모델의 출력값과 `criterion`을 비교하여 loss를 계산하고 파라미터를 업데이트 하게 됩니다. <br>
이때, 본 코드에서 criterion은 `criterion=LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)`로 적용되었습니다. 
LabelSmoothing은 0 또는 1로 표현되는 one-hot-representation인 hard target을 학습하는 것이 아니라 (0,1) 사이 값으로 smoothing을 실시한 soft target을 예측하고자 하는 것입니다.


# 코드 실행 결과
- epoch 총 10회 실행한 결과, epoch가 진행될수록 train과 eval 모두에서 loss가 감소하는 것을 확인할 수 있습니다. <br>
**epoch_result.img**


<Transformer 구조와 관련된 이미지는 transformer 논문 figure와 [Jay Alammar's blog](https://jalammar.github.io/illustrated-transformer/)에서 참고하였습니다.> 
