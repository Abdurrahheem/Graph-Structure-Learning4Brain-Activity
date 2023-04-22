class Config(object):

    ## data params
    N_samples               = 10000
    N_rois                  = 10
    classes                 = 2
    adj_mat_threshold       = 0.2

    ## training params
    batch_size              = 128
    epoch                   = 100
    lr                      = 0.001
    device                  = "cpu"

    ## model params
    hidden_dim              = 32
    model_name              = "GCN"
