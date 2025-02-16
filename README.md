# Image Captioning with Encoder-Decoder Network

## Project Overview
This project implements an image captioning model from scratch using an Encoder-Decoder architecture. The model generates textual descriptions for images by encoding image features using a CNN-based encoder and decoding captions using an LSTM-based decoder.

## Dataset
**Dataset Used:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

Flickr8k consists of 8,000 images, each with five manually written captions. The dataset is preprocessed by tokenizing captions and adding special tokens:
- `<sos>` (Start of Sentence) at the beginning of each caption.
- `<eos>` (End of Sentence) at the end of each caption.

A custom tokenizer was created to encode the captions into a dictionary for training.

## Model Architecture

### **Encoder**
- A **pretrained ResNet-152** is used as the encoder.
- The last fully connected layer is removed to extract image features.
- The **EncoderCNN** processes images to obtain a feature vector of shape `(1, 2048)`.

### **Decoder**
- The **decoder** is built from scratch.
- It consists of:
  1. A **linear layer** to reshape the image feature vector.
  2. A **single-layer LSTM** with a hidden size of `512`.
  3. A **word embedding layer** that processes the input tokenized captions.
  4. During training, **teacher forcing** is applied: the correct token is fed as input to the LSTM.
  5. During inference, predicted tokens are recursively passed as input until `<eos>` is generated.

## Training and Evaluation
The model is trained using cross-entropy loss with teacher forcing to ensure efficient learning.

### **Evaluation Metrics**
Performance is evaluated using standard image captioning metrics:

| Metric  | Score  |
|---------|--------|
| BLEU-1  | 0.484  |
| BLEU-2  | 0.291  |
| BLEU-3  | 0.170  |
| BLEU-4  | 0.098  |
| METEOR  | 0.160  |
| ROUGE-L | 0.369  |
| CIDEr   | 0.247  |
| SPICE   | 0.105  |

## Limitations & Future Improvements
While the model performs reasonably well, there is significant room for improvement:
- **Improving the decoder:** A more sophisticated decoder such as Transformer-based architectures could enhance text generation quality.
- **Larger training set:** Using a larger dataset could improve the model's generalization.
- **Better word embeddings:** Pretrained embeddings (e.g., GloVe, Word2Vec) could improve language fluency.

## Installation & Usage
### **Prerequisites**
- Python 3.8+
- PyTorch
- Torchvision
- NLTK
- NumPy

## Conclusion
This project successfully implements an Encoder-Decoder model for image captioning, demonstrating the effectiveness of CNN-LSTM architectures for vision-to-language tasks. Further enhancements can lead to improved performance and more natural caption generation.

