# Seqence-to-sequence learning for generating financial reports
The project is implemented by the core team of Data Science at Unibit. Our goal is to convert the relational data to financial report. The model we use is Recurrent Neural Network(RNN). The framework is state-of-the-art encoder-decoder with attention and copy mechanism. The implementation is based on tensorflow 1.7 and anaconda environment.

# Model overview
data2report is a nature language generation task that is aiming for helping financial journalist reduce the redundant works and make them focus on the creative writing. Besides, by providing automated reports, we try to induce users to use our valuable data at the backend.        
We use encoder-decoder architecture as our framework. In the first place, we encode our financial table by taking columns name and value of cells thus convert table to the sequence of vectors. Then we fit the embedded table to the RNN cell and store the information of last state in the encoder phase which will be one of input for the decoder phase. As for the decoding phase, hybrid attention mechanism which contains word level attention and feature level attention is proposed to model. By using the attention mechanism, the model can understand what column and value to pick to describe in the report.      

## Dependencies
The code is implemented under ananconda environment and tensorflow 1.7.    
In order to run the file, please install anaconda3 and tensorflow 1.7 with corresponding GPU drives first and then create conda environment for running the code.

## Data
The data for evaluation can be split to X and Y. X is the input information which contains two parts. One is the raw stock price data we scraped from Yahoo finance, the other is the technical indicators that we calculated from the raw data. Y is the financial reports about technical movement of certain stock.

# How to run the code
1. python data_preprocess_1    
2. python data_preprocess_2    
3. python Main.py(for train default, change mode to 'test' to evaluate model)
