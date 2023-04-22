class Config(object):

    ## data params
    N_samples               = 10000
    N_roi                   = 10
    class_num               = 2
    adj_mat_threshold       = 0.2

    ## training params
    batch_size              = 128
    epoch                   = 50
    lr                      = 0.001

    ## model params
    hidden_dim              = 32
    model_name              = "GCN"
