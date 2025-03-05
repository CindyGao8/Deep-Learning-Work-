# Deep-Learning-Work
## Multi-Class Classification on Shuttle Dataset using Decision Trees and Boosting
  This project focuses on multi-class classification using the Shuttle dataset, which consists of 58,000 instances, 10 features (9 numerical and 1 categorical), and 7 output classes. The dataset is imbalanced, as observed in the class distribution plot.
  The study implements Decision Trees and Gradient Boosting Classifiers to analyze and classify data efficiently. Gradient Boosting generally outperforms Decision Trees, but class imbalance remains a challenge. Future improvements could include Random Forests, XGBoost, and resampling techniques like SMOTE to enhance model performance.
## Softmax Derivatives and One-Hot Encoding for Multi-Class Classification 
  This project explores the **derivatives of the softmax function**, essential for backpropagation in multi-class classification models. The derivation includes mathematical steps for computing gradients and their impact on loss optimization.  
  Additionally, the project implements **One-Hot Encoding** for categorical target variables using **Scikit-learn** and applies **data preprocessing techniques** for multi-class classification. The study sets the foundation for understanding **logistic regression extensions, neural network training, and deep learning applications**.
## Parameter Calculation and Memory Estimation for AlexNet & VGG19  
This project focuses on **calculating the number of parameters and memory usage** in deep learning models, specifically **AlexNet and VGG19**. It involves detailed computations for each layer, considering filter sizes, strides, and padding.  

Additionally, a structured table is created to summarize **activation units and parameter counts** at each layer in VGG19. The study helps in understanding the computational complexity of CNN architectures and optimizing model selection based on hardware constraints.

## Siamese Network for Image Similarity with PyTorch  
This project implements a **Siamese Network** using **PyTorch** to perform image similarity tasks. The dataset is preprocessed and structured to create paired inputs for the model. The project includes:  

- **Data Loading & Preprocessing**: Uses PyTorch's `Dataset` class to handle image pairs.  
- **Siamese Network Architecture**: Designed to learn feature embeddings for comparing image similarity.  
- **Visualization & Evaluation**: Uses Matplotlib and PIL for inspecting images and model performance.  

The implementation explores how **contrastive learning** and **deep learning** can be applied to tasks such as **face verification, object matching, and anomaly detection**.

## Sentiment Analysis using RNN, LSTM, and GRU 
This project focuses on **sentiment analysis** using the **IMDB dataset**, implementing different recurrent neural network architectures to compare their performance. The dataset consists of **50,000 movie reviews**, and the models are trained to classify sentiments as positive or negative. The project includes:  

- **Data Preprocessing**: Uses text vectorization techniques to convert raw text into numerical format.  
- **RNN, LSTM, and GRU Implementation**: Builds and compares these architectures using PyTorch.  
- **Training & Evaluation**: Analyzes model performance on sentiment classification tasks.  

The project provides insights into the effectiveness of **Recurrent Neural Networks (RNNs)** and their advanced variants (**LSTM & GRU**) for **natural language processing (NLP) tasks**.

## Question Answering using BERT
This project implements a **BERT-based question-answering system** using the **BertForQuestionAnswering** model and **BertTokenizer**. The goal is to extract answers from a given paragraph based on input questions. The project includes:  

- **Pretrained BERT Model**: Uses `bert-large-uncased-whole-word-masking-finetuned-squad` for question answering.  
- **Text Processing**: Tokenizes and encodes input text using **Hugging Face Transformers**.  
- **Inference and Evaluation**: Extracts relevant answers from the given paragraph using BERT’s attention mechanism.  

This project explores how **pretrained NLP models** can be leveraged for **question answering tasks**, demonstrating the power of **transformer-based architectures** in natural language understanding.

## Interpretability and Feature Selection in Machine Learning  
This project explores the importance of **model interpretability** and **feature selection** in machine learning. The goal is to enhance model transparency, trust, and performance by identifying the most relevant features. The project includes:  

- **Model Interpretability**: Discusses why explainability is crucial for debugging, trust, and compliance with regulations.  
- **Feature Selection**: Implements the `autofeat` library to automatically select the most relevant features for a given dataset.  
- **Diabetes Dataset Analysis**: Uses **Scikit-learn’s diabetes dataset** to demonstrate feature selection techniques.  

This project highlights how **feature selection** and **model explainability** improve decision-making and model reliability in real-world applications.
## Reinforcement Learning and Cloud-Based Model Training 
This project explores key concepts in **reinforcement learning** and **cloud-based model training** using TensorFlow. It covers:  

- **Episodic vs. Continuous Tasks**: Discusses the differences between episodic and continuous reinforcement learning problems, with real-world examples.  
- **Reinforcement Learning Applications**: Examines how decisions impact task progression over time.  
- **Cloud-Based Training**: Implements an **image classification model** using **TensorFlow (v2.14.1)** and cloud services like **Google Cloud AI** and **AzureML** for distributed training.  

This project highlights the significance of **reinforcement learning paradigms** and the efficiency of **scalable cloud-based training for deep learning models**.

## Fine-Tuning a Pretrained MobileNet-SSD Model 
This project focuses on **testing and fine-tuning a pretrained MobileNet-SSD model** for object detection. The implementation follows these steps:  

- **Dataset Preparation**: Uses the **VOC dataset** for training and testing.  
- **Pretrained Model Loading**: Loads a **MobileNet-SSD model** (`mobilenet-v1-ssd-mp-0_675.pth`) for transfer learning.  
- **Fine-Tuning and Evaluation**: Adjusts model parameters to improve object detection performance.  

This project demonstrates how to **fine-tune object detection models** using **transfer learning**, enhancing performance on custom datasets.
