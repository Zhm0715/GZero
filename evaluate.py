import os 
import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--A', type=str,
            default='baselines/A_weights_22969440.ckpt')
    parser.add_argument('--B', type=str,
            default='baselines/A_weights_22969440.ckpt')
    parser.add_argument('--C', type=str,
            default='baselines/A_weights_22969440.ckpt')
    parser.add_argument('--D', type=str,
            default='baselines/A_weights_22969440.ckpt')
    # parser.add_argument('--A', type=str,
    #         default='random')
    # parser.add_argument('--B', type=str,
    #         default='random')
    # parser.add_argument('--C', type=str,
    #         default='random')
    # parser.add_argument('--D', type=str,
    #         default='random')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    evaluate(args.A,
             args.B,
             args.C,
             args.D,
             args.eval_data,
             args.num_workers)
