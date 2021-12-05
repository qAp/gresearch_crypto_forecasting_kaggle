
# Challenge Logbook

## 2021-12-04
Reviewing Yam Peleg's notebook based on the winning solution for Jane Street challenge: https://www.kaggle.com/yamqwe/1st-place-of-jane-street-adapted-to-crypto.

1. Each asset has its own separate model.  Could try a single model for all assets.
2. In the encoder, the standard deviation of the Gaussian Noise layer is equal to the dropout rate for the Dropout layer.
3. The amount of training and data used is the bare minimum, as is the size of the neural network.
4. In tf, how is the batch size passed to the training?

TWML Meet:  
1. One idea is to predict the closing price for the next 16 time steps, then compute from that the target, instead of directly have the model predict the target.  
2. Some approaches allow training for all assets at the same time --- multivariate series models.  But different assets might not be all available at all time stamps, so it's not clear if this can be handled by such methods.




