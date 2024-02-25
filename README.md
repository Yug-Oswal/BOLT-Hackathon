# BOLT-Hackathon
A repository containing my team's Machine Learning solutions and projects for the BOLT Hackathon. 

## Models Trained/Fine-tuned/Utilised: 
1. Gemma 2B
2. BERT (Preprocessor + Encoder Layers of the network till the 'pooled_output' layer) + Custom Layers (Dropout, Dense) layers + Classification (Dense 'softmax') layer
3. DistilBERT (Preprocessor + Backbone layers of the network + Custom Layers (GlobalAveragePooling, Dropout, Dense) + Classification layer
4. Custom Model: Embedding + RNN Layers (LSTM, GRU) + Spatial Dropout Layers + Dropout, Dense Layers + Classification layer 

## Intel Technologies Utilised: 
1. intel-tensorflow: The Intel Tensorflow library was used for three tasks: Training of custom Recurrent Neural models, Fine-tuning of Language Models, and Fine-tuning of Preprocessing-Encoder-Fully Connected-Output layer like models. The intel-tensorflow library proved to be very easy to access and use, and I had almost no hiccups after the initial setting up phase of the library. The library provides beautifully similar API functionality to that of the original tensorflow and comes as a part of the Intel AI toolkit. Additional similar libraries explored: Intel HPC, Intel Neural Compressor, Intel OpenVINO toolkit.
2. modin (for pandas): We dissected and used the Google Go-Emotions Dataset consisted of approx. 210,000 texts and 28 emotion categories as labels for the same. Using modin for pandas provided once again, a very easy-to-switch-over API and boosted our data cleaning processes by significantly.
3. sklearnex: Scikit-learn was used for scoring different metrics on the dataset and in data pre-processing. The intel extension on sklearn took very little to no effort to setup, hoewver, ended up providing an almost seamless speedup of performance during data cleaning.
4. Intel Development Console (IDC): The IDC was the cornerstone of our project. It provided an easy-to-use interface and was where the entire project was deployed and hosted. The Intel VM provided us with low latencies on requests and much higher response times on LLM inference for the VM-hosted Gemma 2B model being used in the project.
