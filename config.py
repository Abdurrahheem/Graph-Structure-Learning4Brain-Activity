class Config(object):

    ## data params
    N_samples               = 10000
    N_rois                  = 100
    classes                 = 2
    adj_mat_threshold       = 0.2

    ## training params
    batch_size              = 30
    epoch                   = 100
    lr                      = 0.0001
    device                  = "cuda:0"
    weight_decay            = 0.0001

    ## model params
    hidden_dim              = 30
    model_name              = "GCN"
