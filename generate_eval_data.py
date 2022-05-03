import argparse
import pickle
import numpy as np

deck = []
for i in range(0, 52):
    deck.append(i)


def get_parser():
    parser = argparse.ArgumentParser(description='DouZero: random data generator')
    parser.add_argument('--output', default='eval_data', type=str)
    parser.add_argument('--num_games', default=1, type=int)
    return parser
    
def generate():
    _deck = deck.copy()
    np.random.shuffle(_deck)
    for _ in range(len(_deck)):
        if _deck[_] == 26:
            tmp = _deck[0]
            _deck[0] = _deck[_]
            _deck[_] = tmp
    card_play_data = {
        'A': _deck[:13],
        'B': _deck[13:26],
        'C': _deck[26:39],
        'D': _deck[39:52]
    }
    for key in card_play_data:
        card_play_data[key].sort()
    return card_play_data


if __name__ == '__main__':
    flags = get_parser().parse_args()
    output_pickle = flags.output + '.pkl'

    print("output_pickle:", output_pickle)
    print("generating data...")

    data = []
    for _ in range(flags.num_games):
        data.append(generate())

    print("saving pickle file...")
    with open(output_pickle,'wb') as g:
        pickle.dump(data,g,pickle.HIGHEST_PROTOCOL)




