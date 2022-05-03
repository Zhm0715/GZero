import multiprocessing as mp
import pickle
#
from douzero.env.game import GameEnv
from .deep_show_card import DeepShowAgent

def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ['A', 'B', 'C', 'D']:
        if card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent
            players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players



def mp_simulate(card_play_data_list, card_play_model_path_dict, q):
    game_log_file_path = "GameLog.txt"
    players = load_card_play_models(card_play_model_path_dict)
    env = GameEnv(players)
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data, True, game_log_file_path)
        show_info = {"A": [], "B": [], "C": [], "D": []}
        for position in ['A', 'B', 'C', 'D']:
            show_model = DeepShowAgent(position, "baselines/showA_weights_22969440.ckpt")
            act = show_model.act(show_info, card_play_data[position])
            show_info[position] = act
            env.update_show_card(position, act)
            with open(game_log_file_path, "a") as f:
                f.write("S " + position + " " + str(",".join(map(str, act))) + "\n")
        while not env.game_over:
            env.step(True, game_log_file_path)
    #     env.reset()
    #
    q.put((env.num_wins['A'],
           env.num_wins['B'],
           env.num_wins['C'],
           env.num_wins['D'],
           env.num_scores['A'],
           env.num_scores['B'],
           env.num_scores['C'],
           env.num_scores['D']
         ))

def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker

def evaluate(A, B, C, D, eval_data, num_workers):

    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'A': A,
        'B': B,
        'C': C,
        'D': D
    }

    num_A_wins = 0
    num_B_wins = 0
    num_C_wins = 0
    num_D_wins = 0
    num_A_scores = 0
    num_B_scores = 0
    num_C_scores = 0
    num_D_scores = 0

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_paly_data in card_play_data_list_each_worker:
        p = ctx.Process(
                target=mp_simulate,
                args=(card_paly_data, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_A_wins += result[0]
        num_B_wins += result[1]
        num_C_wins += result[2]
        num_D_wins += result[3]
        num_A_scores += result[4]
        num_B_scores += result[5]
        num_C_scores += result[6]
        num_D_scores += result[7]
    print(num_A_scores)
    print(num_A_wins)
    print(num_B_scores)
    print(num_B_wins)
    print(num_C_scores)
    print(num_C_wins)
    print(num_D_scores)
    print(num_D_wins)
