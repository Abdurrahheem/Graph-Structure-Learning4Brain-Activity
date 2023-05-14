class Config(object):
    ## LogRegression for comparision
    LogRegression           = False

    ## sythetic data params
    dataset                 = "synthetic"
    N_samples               = 10000
    N_rois                  = 100
    classes                 = 2

    ## cobre dataset parameters
    dataset                 = "cobre"
    classes                 = 3
    N_rois                  = 116

    ## Adjacency matrix parameters
    adj_mat_threshold       = 0.2

    ## training params
    seed                    = 1234
    batch_size              = 30
    epoch                   = 100
    lr                      = 0.0001
    weight_decay            = 0.0001
    device                  = "cuda:0"
    val_size                = 0.2

    ## model params
    hidden_dim              = 30
    model_name              = "GCN"
