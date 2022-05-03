from collections import Counter
import numpy as np

from douzero.env.game import GameEnv
from itertools import combinations
from torch import tensor

deck = []
for i in range(0, 52):
    deck.append(i)

"""
self._env = GameEnv()
"""


class Env:
    """
    斗地主代理包装类
    Doudizhu multi-agent wrapper
    """

    def __init__(self, objective):
        """
        Objective is wp/adp/logadp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.show_card = {'A': [],
                          'B': [],
                          'C': [],
                          'D': []}
        self.objective = objective

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['A', 'B', 'C', 'D']:
            self.players[position] = DummyAgent(position)

        # Initialize the internal environment
        self._env = GameEnv(self.players)

        self.infoset = None

    def reset(self, model, device, flags=None):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()
        # Randomly shuffle the deck
        # deck is the hand cards
        _deck = deck.copy()
        # 找到黑桃2所在位置并设置为A
        np.random.shuffle(_deck)
        # 找到梅花2 和0位置交换
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
        self.show_card = {'A': [],
                          'B': [],
                          'C': [],
                          'D': []}
        for key in card_play_data:
            card_play_data[key].sort()
        all_show_obs = dict()

        for _pos in ['A', 'B', 'C', 'D']:
            show_obs = _get_obs_show_card(_pos, self.show_card, card_play_data[_pos])
            agent_output = model.forward('show' + _pos, tensor(tensor([]), device=device),
                                         tensor(show_obs['x_batch'], device=device), flags=flags)
            _action_idx = int(agent_output['action'].cpu().detach().numpy())
            show_card_list = show_obs['legal_actions'][_action_idx]
            self.show_card[_pos].extend(show_card_list)
            self._env.update_show_card(_pos, show_card_list)
            obs = {'obs': show_obs, 'action': show_card_list}
            all_show_obs[_pos] = obs
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset
        return get_obs(self.infoset), all_show_obs

    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = {
            'A': 0.0,
            'B': 0.0,
            'C': 0.0,
            'D': 0.0,
        }
        if self._game_over:
            done = True
            reward = self._get_reward()  # score
            obs = None
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        score = self._env.num_scores
        return score

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over


class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """

    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action


def get_obs(infoset):
    """
    This function obtains observations with imperfect information
    from the infoset. It has three branches since we encode
    different features for different positions.

    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `position` is a string that can be landlord/landlord_down/landlord_up

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `legal_actions` is the legal moves

    `x_no_action`: the features (exluding the hitorical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """
    if infoset.player_position in ['A', 'B', 'C', 'D']:
        return _get_obs_player(infoset, infoset.player_position)
    else:
        raise ValueError('')


def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot


def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    # TODO modify the matrix size, WARNING!!!
    if len(list_cards) == 0:
        return np.zeros(52, dtype=np.int8)
    matrix = np.zeros(52, dtype=np.int8)
    for j in list_cards:
        matrix[j] = 1
    return matrix


def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    """
    # TODO Modify the matrix size, WARNING!!! 54->52
    action_seq_array = np.zeros((len(action_seq_list), 52))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    # TODO Modify the matrix size, WARNING!!! 162->156
    action_seq_array = action_seq_array.reshape(5, 156)
    return action_seq_array


def _process_action_seq(sequence, length=15):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


# get the all type of show card
def _gen_show_card(hand_cards):
    legal_show_card = []
    all_card_can_show = []
    if 23 in hand_cards:
        all_card_can_show.append(23)
    if 12 in hand_cards:
        all_card_can_show.append(12)
    if 34 in hand_cards:
        all_card_can_show.append(34)
    if 48 in hand_cards:
        all_card_can_show.append(48)
    show_card_num = len(all_card_can_show)
    legal_show_card.append([])  # show nothing
    for num in range(1, show_card_num + 1):
        single_combine = list(combinations(all_card_can_show, num))
        for _tuple in single_combine:
            legal_show_card.append(list(_tuple))
    return legal_show_card

def _get_sort(position):
    if position == 'A':
        return ['B', 'C', 'D']
    if position == 'B':
        return ['C', 'D', 'A']
    if position == 'C':
        return ['D', 'A', 'B']
    if position == 'D':
        return ['A', 'B', 'C']

def _get_obs_show_card(position, show_info, hand_cards):
    num_legal_actions = len(_gen_show_card(hand_cards))
    other_show_card = []
    other_show_card_batch = []
    all_show_card_info = []
    hand_card = _cards2array(hand_cards)
    hand_card_batch = np.repeat(hand_card[np.newaxis, :],
                                num_legal_actions, axis=0)
    pos_sort = _get_sort(position)
    for _pos in pos_sort:
        _other_show = _cards2array(show_info[_pos])
        other_show_card.append(_other_show)
        other_show_card_batch.append(np.repeat(_other_show[np.newaxis, :],
                                               num_legal_actions, axis=0))
        all_show_card_info.extend(show_info[_pos])

    all_show_card = _cards2array(all_show_card_info)

    all_show_card_batch = np.repeat(all_show_card[np.newaxis, :],
                                    num_legal_actions, axis=0)

    my_show_card = _cards2array(show_info[position])
    my_show_card_batch = np.repeat(my_show_card[np.newaxis, :],
                                   num_legal_actions, axis=0)

    legal_actions = _gen_show_card(hand_cards)
    for j, action in enumerate(legal_actions):
        my_show_card_batch[j, :] = _cards2array(action)

    x_batch = np.hstack((hand_card_batch,
                         other_show_card_batch[0],
                         other_show_card_batch[1],
                         other_show_card_batch[2],
                         all_show_card_batch,
                         my_show_card_batch))
    x_no_action = np.hstack((hand_card,
                             other_show_card[0],
                             other_show_card[1],
                             other_show_card[2],
                             all_show_card))

    obs = {
        'position': "",
        'x_batch': x_batch.astype(np.float32),
        'legal_actions': legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        "show_card_batch": my_show_card_batch.astype(np.int8)
    }
    return obs


def _get_obs_player(infoset, position):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    """

    """
    num_legal_actions = len(infoset.legal_actions)

    my_hand_cards = _cards2array(infoset.player_hand_cards)

    my_hand_cards_batch = np.repeat(my_hand_cards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_hand_cards = _cards2array(infoset.other_hand_cards)

    other_hand_cards_batch = np.repeat(other_hand_cards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    pos_sort = _get_sort(position)
    loop_move_card = []
    for _pos in pos_sort:
        loop_move_card.append(infoset.loop_move[_pos])
    loop_move = _cards2array(loop_move_card)
    loop_move_batch = np.repeat(loop_move[np.newaxis, :],
                                num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_hand_cards_batch.shape)

    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)
    other_played_hand_cards_info = []
    other_played_hand_cards_info_batch = []
    pos_sort = _get_sort(position)
    for _pos in pos_sort:
        _played_cards = _cards2array(infoset.played_cards[_pos])
        other_played_hand_cards_info.append(_played_cards)
        _played_cards_batch = np.repeat(
            _played_cards[np.newaxis, :],
            num_legal_actions, axis=0)
        other_played_hand_cards_info_batch.append(_played_cards_batch)
    # win card
    my_win_card_info = _cards2array(infoset.won_cards[position])
    my_win_card_info_batch = np.repeat(
        my_win_card_info[np.newaxis, :],
        num_legal_actions, axis=0)
    other_win_card_info = []
    other_win_card_info_batch = []
    for _pos in pos_sort:
        _played_cards = _cards2array(infoset.won_cards[_pos])
        other_win_card_info.append(_played_cards)
        _played_cards_batch = np.repeat(
            _played_cards[np.newaxis, :],
            num_legal_actions, axis=0)
        other_win_card_info_batch.append(_played_cards_batch)

    # show card information
    my_show_card = _cards2array(infoset.showed_cards[position])
    my_show_card_batch = np.repeat(
        my_show_card[np.newaxis, :],
        num_legal_actions, axis=0
    )
    other_showed_card_info = []
    other_showed_card_info_batch = []
    for _pos in pos_sort:
        _other_showed_cards = _cards2array(infoset.showed_cards[_pos])
        other_showed_card_info.append(_other_showed_cards)
        _other_showed_cards_batch = np.repeat(
            _played_cards[np.newaxis, :],
            num_legal_actions, axis=0
        )
        other_showed_card_info_batch.append(_other_showed_cards_batch)

    x_batch = np.hstack((my_hand_cards_batch,
                         other_hand_cards_batch,
                         my_win_card_info_batch,
                         other_win_card_info_batch[0],
                         other_win_card_info_batch[1],
                         other_win_card_info_batch[2],
                         my_show_card_batch,
                         other_showed_card_info_batch[0],
                         other_showed_card_info_batch[1],
                         other_showed_card_info_batch[2],
                         other_played_hand_cards_info_batch[0],
                         other_played_hand_cards_info_batch[1],
                         other_played_hand_cards_info_batch[2],
                         loop_move_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_hand_cards,
                             other_hand_cards,
                             my_win_card_info,
                             other_win_card_info[0],
                             other_win_card_info[1],
                             other_win_card_info[2],
                             my_show_card,
                             other_showed_card_info[0],
                             other_showed_card_info[1],
                             other_showed_card_info[2],
                             other_played_hand_cards_info[0],
                             other_played_hand_cards_info[1],
                             other_played_hand_cards_info[2],
                             loop_move,
                             ))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
        'position': position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


if __name__ == "__main__":
    # test
    s = _cards2array([1, 2, 3])
    print(s)