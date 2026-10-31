# 課程筆記

## Supervised Learning

給模型一批有標籤資料，讓模型學會從輸入 `x` 預測輸出 `y`。例如給很多圖片與標籤，模型學會分辨新的貓狗圖片。

主要限制是標註資料成本高，而且每個任務通常都需要重新標資料，因此雖然監督式學習有效，但不容易擴展到所有領域。

## Self-supervised Learning

讓機器利用資料本身的結構產生訓練訊號。例如遮住句子中的幾個字，讓模型預測被遮住的內容。

可以把 pre-trained model 想成作業系統，downstream tasks 想成應用程式：先學出通用能力，再遷移到不同任務。

常見例子：

- BERT：Masked Language Modeling
- GPT：Next Token Prediction

## Generative Adversarial Network

GAN 透過 Generator 和 Discriminator 的對抗訓練，讓模型學會生成接近真實資料分佈的樣本。

## Reinforcement Learning

當沒有明確標籤，但能判斷結果好壞時，可以使用強化學習。模型透過 trial and error 學習策略，根據 reward 改進行動。

例子包含 AlphaGo、自駕車、Atari 遊戲。

## Anomaly Detection

模型學習正常資料分佈，當輸入偏離正常分佈時，能判斷「這個樣本不確定或異常」。

## Explainable AI

AI 不只要做對，也要能解釋原因。模型需要指出哪些輸入特徵影響決策，讓人能理解判斷依據。

## Model Attack

模型可能對微小擾動非常敏感。只要對輸入加入人眼幾乎察覺不到的改動，模型就可能被誤導。

## Domain Adaptation

當訓練資料和測試資料分佈差異很大時，模型表現會下降。Domain Adaptation 的目標是讓模型能適應不同資料分佈。

## Network Compression

模型太大時不容易部署到手機或邊緣設備。常見方法包含 pruning、quantization、knowledge distillation。

## Life-long Learning

希望 AI 能持續學習新知識，同時避免忘記舊任務，也就是降低 catastrophic forgetting。

## Meta Learning

Meta Learning 是讓模型學會如何快速學習新任務，也就是 learning to learn。常見應用是 few-shot learning。
