import argparse


def load_arguments(dataset=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset",
                           default="test",
                           type=str,
                           help="the dataset to be attacked")
    
    parser.add_argument("--victim",
                           default="ag",
                           type=str,
                           help="the victim to be attacked")

    # parser.add_argument("--sample_number",
    #                        default=50,
    #                        type=int,
    #                        help="the number of samples")
    # parser.add_argument("--sample_batch_size",
    #                        default=50,
    #                        type=int,
    #                        help="the batch size of sampling")
    # parser.add_argument("--fill_mask_model",
    #                        default="BART",
    #                        type=str,
    #                        help="model for phrase mask filling")


    args = parser.parse_args()
    print(args)
    return args
