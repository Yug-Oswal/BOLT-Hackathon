# BOLT-Hackathon: Intel Track
A repository containing my team's projects and solutions under the BOLT Hackathon under the Intel track.

## Related Repositories
1. Frontend: https://github.com/manaslaud/intel-hack
2. Backend: https://github.com/woaitsAryan/rekindle-backend
3. Machine Learning: https://github.com/Yug-Oswal/BOLT-Hackathon/

## Our Project - Rekindle 
<img src="https://i.imgur.com/An2OprO.jpeg">

## Models Trained/Fine-tuned/Utilised: 
1. Gemma 2B served and run via Ollama
2. Ensemble Retreiver: Ensemble of 2 retrievers: 30% weightage to semantic search (using HuggingFace embeddings + ChromaDB) + 70% weightage to BM25 search 
3. BERT (Preprocessor + Encoder Layers of the network till the 'pooled_output' layer) + Custom Layers (Dropout, Dense) layers + Classification (Dense 'softmax') layer
 ```
preprocessor = hub.KerasLayer(
   'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
encoder = hub.KerasLayer(
   'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/2',
   trainable=True)
txt = tf.keras.layers.Input(shape=(), dtype=tf.string)
x = preprocessor(txt)
x = encoder(x)['pooled_output']
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(Y_train1.shape[1], activation='softmax')(x)
model = tf.keras.Model(inputs=[txt], outputs=x)
optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4)
metrics = [tf.keras.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
model.compile(optimizer, loss, metrics=metrics)
``` 
3. DistilBERT (Preprocessor + Backbone layers of the network + Custom Layers (GlobalAveragePooling, Dropout, Dense) + Classification layer
```
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=128,
)
encoder = keras_nlp.models.DistilBertBackbone.from_preset(
    "distil_bert_base_en_uncased"
)
encoder.trainable = True
txt = tf.keras.layers.Input(shape=(), dtype=tf.string)
x = preprocessor(txt)
x = encoder(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(Y_train1.shape[1], activation='softmax')(x)
model = tf.keras.Model(inputs=[txt], outputs=x)
```
5. Custom Model: Embedding + RNN Layers (LSTM, GRU) + Spatial Dropout Layers + Dropout, Dense Layers + Classification layer
```
input = Input(shape=(MAX_SEQUENCE_LENGTH, ))
x = Embedding(MAX_NB_WORDS, EMBEDDING_DIM)(input)
x = SpatialDropout1D(0.2)(x)
x = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(x)
output = Dense(28, activation='softmax')(x)
model = tf.keras.Model(inputs=input, outputs=output)
```

## Machine Learning Decisions: 
1. Gemma 7B, Mistral 7B, Phi 2.7, GPT 3.5 Turbo, and Gemma 2B were the pre-trained LLMs which were locally hosted and run before the final model was selected. Inference times, low latencies, low frequencies of hallucinations, and high quality responses were the metrics used to determine the model.
2. The Gemma 2B was chosen since other larger models (such as those with 7B params) proved to have larger inference times than expected. While the Gemma model served our purposed best since the target audience of our project lies in aged citizens. There are no expected multiple corrupted requests and unnecessarily long token length prompts from such an audience. Gemma 2B provides better response times (due to loading faster since it has smaller manifests), lower inference times (courtesy of it's smaller size - 2B params), and good quality responses, tested over some hand-curated stories which were prompted to the model.
3. The BERT model was re-trained, with pre-loaded weights. That is, the encoder layers of the network, along with newly added layers underwent backpropagation. This  model was used in a tutorial by Google on the same dataset our model was fine-tuned on and served as a primary benchmark for the same.
4. The best weightages for the retrieval were done via empirical testing. Retrievers more than 2 proved to have a higher latency, and after a certain point even had diminishing or negative returns. The embeddings were stored and persisted using chromadb to reach faster data retrieval times.
5. The Custom Model would have served the best purpose for the task of emotion extraction from text. However, the training of such a model (as shown in the ipynb notebook), even with a relatively basic architecture proved to take up a lot of compute and time, and wasn't worth the returns. Thus, our team turned to fine-tuning models, of which, the BERT model served as a benchmark.

## Intel Technologies Utilised: 
1. Intel Development Console (IDC): The IDC was the cornerstone of our project. It provided an easy-to-use interface and was where the entire project was deployed and hosted. The Intel VM provided us with low latencies on requests and much higher response times on LLM inference for the VM-hosted Gemma 2B model being used in the project.
2. intel-tensorflow: The Intel Tensorflow library was used for three tasks: Training of custom Recurrent Neural models, Fine-tuning of Language Models, and Fine-tuning of Preprocessing-Encoder-Fully Connected-Output layer like models. The intel-tensorflow library proved to be very easy to access and use, and I had almost no hiccups after the initial setting up phase of the library. The library provides beautifully similar API functionality to that of the original tensorflow and comes as a part of the Intel AI toolkit. Additional similar libraries explored: Intel HPC, Intel Neural Compressor, Intel OpenVINO toolkit.
3. modin (for pandas): We dissected and used the Google Go-Emotions Dataset consisting of approx. 210,000 texts and 28 emotion categories as labels for the same. Using modin for pandas provided once again, a very easy-to-switch-over API and boosted our data cleaning processes by significantly.
4. sklearnex: Scikit-learn was used for scoring different metrics on the dataset and in data pre-processing. The intel extension on sklearn took very little to no effort to setup, hoewver, ended up providing an almost seamless speedup of performance during data cleaning.

## Resources: 
1. Google Go-Emotions Dataset: https://blog.research.google/2021/10/goemotions-dataset-for-fine-grained.html - 210,000 texts from Reddit threads, Corpus of 58k words, 27 emotion categories + 1 neutral category
