[EE638] - Implementation of "Acoustic-to-articulatory inversion of dysarthric speech using self-supervised learning"

## Abstract 
Acoustic-to-articulatory inversion (AAI) involves mapping the acoustic speech signal to the articulatory space which specifies the motion of the speech articulators like the lips, jaw, tongue, and velum, in order to understand the speech production dynamics. Neural networks, with signal-processing features like MFCCs, have been widely used for the AAI task. In this work, we perform acousticto- articulatory inversion (AAI) for dysarthric speech using representations from pre-trained self-supervised learning (SSL) models. SSL has been proven beneficial for other spoken language understanding tasks, and hence, we demonstrate the impact of different pre-trained features in the articulatory inversion of dysarthric speech, for the TORGO dataset. In addition, we also condition speaker information using x-vectors to the extracted robust SSL features, as an input, to train a BLSTM network. We experiment with three different AAI training schemes (subject-specific, pooled, and fine-tuned). Experimental results reveal that DeCoAR, in the fine-tuned scheme, achieves a relative improvement of the Pearson Correlation Coefficient (CC) by ∼1.81% and ∼4.56% for healthy controls and patients, respectively, over MFCCs. Overall, we find that SSL networks, like wav2vec, APC, and DeCoAR, that are trained with feature reconstruction or future timestep prediction tasks, perform well across all three AAI training schemes for predicting dysarthric articulatory trajectories.


## Installation 
    $ git clone https://github.com/sarthaxxxxx/AAI-SSL-Dysarthria.git
    $ cd AAI-SSL-Dysarthria/
    $ pip install requirements.txt
    
Set the parameters in config/params.yaml based on the mode of training/inference scheme. Then, 
```
python3 runner.py --config (path to the config file in your system) --gpu "set your cuda device"
```

## License
MIT License

## Contact
  Sarthak Kumar Maharana - email: maharana@usc.edu 
