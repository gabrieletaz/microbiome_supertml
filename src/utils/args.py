import torch
import argparse


# ----- Print Parser -----

'''def print_args(ARGS):
    print('\n'+26*'='+' Configuration '+26*'=')
    for name, var in vars(ARGS).items():
        print('{} : {}'.format(name, var))
    print('\n'+25*'='+' Training Starts '+25*'='+'\n')
'''

# ----- Parser -----

def parser():
    PARSER = argparse.ArgumentParser(description='Training parameters.')

    PARSER.add_argument('--dataset', default='IBD', type=str,
                        choices=['IBD', 'Obesity', 'Cirrhosis', 'CT2D', 'WT2D', 'Colorectal', 'pretrain'], help='Dataset.')
    
    PARSER.add_argument('--model', default='CNN_opt', type=str,
                        choices=['CNN_opt', 'MLP_opt'], help='Model.')

    PARSER.add_argument('--batch_size', default=8, type=int,
                        help='Batch size.')

    PARSER.add_argument('--device', default='cuda:0', type=str,
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
                        help='Device to run the experiment.')
    
    PARSER.add_argument('--precision', default=5, type=int,
                        help='Precision of the printed data')

    PARSER.add_argument('--font_size', default=0.21, type=float,
                        help='Precision of the printed data')
    
    PARSER.add_argument('--img_type', default='digits', type=str,
                        choices=['digits', 'tabular'],
                        help='digits or rectangles')

    PARSER.add_argument('--aug', default='False', type=str,
                        choices=['RandomErasing', 'CellDropout', 'RandGauss', 'RandCoarseDrop',
                                 'RandCoarseShuffle', 'RandBiasField', 'RandElastic', 'False', 'RandZoom', 'RandRotate', 'RandFlip'],
                        help='digits or rectangles')

    PARSER.add_argument('--exp_name', default='test', type=str)
    
    ARGS = PARSER.parse_args()

    if ARGS.device is None:
        ARGS.device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")

    #print_args(ARGS)

    return ARGS


args = parser()

if __name__ == "__main__":
    pass
