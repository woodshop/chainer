#!/ust/bin/env python

from chainer import FunctionSet, Variable, optimizers
import chainer.functions as F
import numpy as np
import cPickle

# Adapted from Lasagne
# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 55
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 1000

MIN_LENGTH = 10
MAX_LENGTH = 30
# # Number of units in the hidden (recurrent) layer
N_HIDDEN = 25
# # Number of training sequences in each batch
N_BATCH = 20

def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    '''
    Generate a batch of sequences for the "add" task, e.g. the target for the
    following

    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``

    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.

    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    n_batch : int
        Number of samples in the batch.

    Returns
    -------
    X : np.ndarray
        Input to the network, of shape (n_batch, max_length, 2), where the last
        dimension corresponds to the two sequences shown above.
    y : np.ndarray
        Correct output for each sample, shape (n_batch,).
    mask : np.ndarray
        A binary matrix of shape (n_batch, max_length) where ``mask[i, j] = 1``
        when ``j <= (length of sequence i)`` and ``mask[i, j] = 0`` when ``j >
        (length of sequence i)``.

    References
    ----------
    .. [1] Hochreiter, Sepp, and Jurgen Schmidhuber."Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.

    .. [2] Sutskever, Ilya, et al. "On the importance of initialization and
    momentum in deep learning." Proceedings of the 30th international
    conference on machine learning (ICML-13). 2013.
    '''
    # Generate X - we'll fill the last dimension later
    X = np.concatenate(
        [np.random.uniform(size=(n_batch, max_length, 1)) +
         1j*np.random.uniform(size=(n_batch, max_length, 1)),
         np.zeros((n_batch, max_length, 1))], axis=-1)

    mask = np.zeros((n_batch, max_length))
    y = np.zeros((n_batch,), np.complex64)
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        # Make the mask for this sample 1 within the range of length
        mask[n, :length] = 1
        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/10), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        # y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
        y[n] = np.prod(X[n,X[n,:,1].astype(np.bool),0])
        # Center the inputs and outputs
        X -= X.reshape(-1, 2).mean(axis=0)
        y -= y.mean()
        X /= X.reshape(-1, 2).std(axis=0)
        y /= y.std()
        return (X.astype(np.complex64), y.astype(np.complex64), mask)


def forward(model, X, mask, **kwargs):
    state = {'hid_fwd': Variable(
        np.zeros((N_BATCH, N_HIDDEN)).astype(np.complex64), **kwargs),
             'c_fwd': Variable(
        np.zeros((N_BATCH, N_HIDDEN)).astype(np.complex64), **kwargs),
             'hid_bwd': Variable(
        np.zeros((N_BATCH, N_HIDDEN)).astype(np.complex64), **kwargs),
             'c_bwd': Variable(
        np.zeros((N_BATCH, N_HIDDEN)).astype(np.complex64), **kwargs)}
    for i in xrange(X.shape[1]):
        j = X.shape[1] - i - 1
        X_var_fwd = Variable(X[:,i], **kwargs)
        X_var_bwd = Variable(X[:,j], **kwargs)
        hid_fwd_in = (model.in_to_hid_fwd(X_var_fwd) +
                      model.hid_to_hid_fwd(state['hid_fwd']))
        c_fwd, hid_fwd = lstm(state['c_fwd'], hid_fwd_in)
        c_fwd = mask[:,[i]] * c_fwd + (1-mask[:,[i]]) * state['c_fwd']
        hid_fwd = mask[:,[i]] * hid_fwd + (1-mask[:,[i]]) * state['hid_fwd']
        hid_bwd_in = (model.in_to_hid_bwd(X_var_bwd) +
                      model.hid_to_hid_bwd(state['hid_bwd']))
        c_bwd, hid_bwd = lstm(state['c_bwd'], hid_bwd_in)
        c_bwd = mask[:,[j]] * c_bwd + (1-mask[:,[j]]) * state['c_bwd']
        hid_bwd = mask[:,[j]] * hid_bwd + (1-mask[:,[j]]) * state['hid_bwd']
        output = F.tanh(model.hid_to_out(F.concat((hid_fwd, hid_bwd))),
                        cplx=True)
        state = {'hid_fwd': hid_fwd, 'c_fwd': c_fwd,
                 'hid_bwd': hid_bwd, 'c_bwd': c_bwd}
    return output

def run():
    best = np.inf
    wscale = 0.1
    model = FunctionSet(
        in_to_hid_fwd=F.Linear(2, 4 * N_HIDDEN, wscale=wscale,
                               cplx=True),
        hid_to_hid_fwd=F.Linear(N_HIDDEN, 4* N_HIDDEN, wscale=wscale,
                                cplx=True),
        in_to_hid_bwd=F.Linear(2, 4 * N_HIDDEN, wscale=wscale,
                               cplx=True),
        hid_to_hid_bwd=F.Linear(N_HIDDEN, 4 * N_HIDDEN, wscale=wscale,
                                cplx=True),
        hid_to_out=F.Linear(2*N_HIDDEN, 1, wscale=wscale,
                            cplx=True))
    optimizer = optimizers.SGD(lr=LEARNING_RATE, cplx=True)
    optimizer.setup(model)
    X_val, y_val, mask_val = gen_data()
    y_val_var = Variable(y_val[:, np.newaxis], volatile=True)


    for epoch in xrange(NUM_EPOCHS):
        model_out = forward(model, X_val, mask_val, volatile=True)
        loss = F.mean_squared_error(model_out, y_val_var, cplx=True)
        print("{}\t{}".format(epoch, loss.data.real))
        if loss.data.real < best:
            best = loss.data.real
            save_model(model)
        for _ in xrange(EPOCH_SIZE):
            X, y, mask = gen_data()
            y_var = Variable(y[:, np.newaxis])
            model_out = forward(model, X, mask, volatile=False)
            loss = F.mean_squared_error(model_out, y_var, cplx=True)
            optimizer.zero_grads()
            loss.backward()
            loss.unchain_backward()
            optimizer.clip_grads(GRAD_CLIP)
            optimizer.update()
    model_out = forward(model, X_val, mask_val, volatile=True)
    loss = F.mean_squared_error(model_out, y_val_var, cplx=True)
    if loss.data.real < best:
        best = loss.data.real
        save_model(model)
    print("{}\t{}".format(epoch+1, loss.data.real))

def save_model(model):
    with open('/home/asarroff/tmp/product_model.pkl', 'w') as f:
        cPickle.dump(model.parameters, f, -1)

if __name__ == "__main__":
    lstm = F.CplxLSTM()
    run()
