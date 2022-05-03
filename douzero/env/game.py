from copy import deepcopy

# score_card = {23: -100, 48: 100,
#               3: -10, 4: -10, 5: -10, 6: -10, 7: -10, 8: -10,
#               9: -20, 10: -30, 11: -40, 12: -50}

score_card = {23: -1, 48: 1, 0: 0, 1: 0, 2: 0, 3: -0.1, 4: -0.1, 5: -0.1, 6: -0.1, 7: -0.1, 8: -0.1, 9: -0.2, 10: -0.3, 11: -0.4,
              12: -0.5}


def contain(win_card, target_card):
    for _card in target_card:
        if not [_card] in win_card:
            return False
    return True


class GameEnv(object):
    """
    配置游戏环境
    """

    def __init__(self, players):

        self.card_play_action_seq = []

        self.game_over = False

        # which one play
        self.acting_player_position = None

        self.players = players

        # the card type first loop and whether show card can be play
        self.first_play = [False, False, False, False]

        # infoset中的引用
        self.played_cards = {'A': [],
                             'B': [],
                             'C': [],
                             'D': []}

        # infoset中的引用
        self.won_cards = {'A': [],
                          'B': [],
                          'C': [],
                          'D': []}

        self.showed_card = {'A': [],
                            'B': [],
                            'C': [],
                            'D': []}

        self.loop_move = {'A': [],
                          'B': [],
                          'C': [],
                          'D': []}

        self.all_showed_card = []

        self.num_wins = {'A': 0,
                         'B': 0,
                         'C': 0,
                         'D': 0}

        self.num_scores = {'A': 0,
                           'B': 0,
                           'C': 0,
                           'D': 0}

        self.info_sets = {'A': InfoSet('A'),
                          'B': InfoSet('B'),
                          'C': InfoSet('C'),
                          'D': InfoSet('D')}

    def card_play_init(self, card_play_data, write_file=False, game_log_file_path=""):
        if write_file:
            with open(game_log_file_path, "w+") as f:
                f.seek(0)
                f.truncate()
            for _pos in ['A', 'B', 'C', 'D']:
                with open(game_log_file_path, "a") as f:
                    f.write("H " + _pos + " " + str(",".join(map(str, card_play_data[_pos]))) + "\n")
        self.info_sets['A'].player_hand_cards = \
            card_play_data['A']
        self.info_sets['B'].player_hand_cards = \
            card_play_data['B']
        self.info_sets['C'].player_hand_cards = \
            card_play_data['C']
        self.info_sets['D'].player_hand_cards = \
            card_play_data['D']
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def game_done(self):
        if len(self.info_sets['A'].player_hand_cards) == 0 and \
                len(self.info_sets['B'].player_hand_cards) == 0 and \
                len(self.info_sets['C'].player_hand_cards) == 0 and \
                len(self.info_sets['D'].player_hand_cards) == 0:
            # if one of the three players discards his hand,
            # then game is over.
            self.update_num_wins_scores()
            self.compute_player_utility()
            self.game_over = True

    def compute_player_utility(self):
        win_player_cnt = 0
        for _pos in ['A', 'B', 'C', 'D']:
            if self.num_scores[_pos] > 0:
                win_player_cnt += 1
                self.num_wins[_pos] = 1
            else:
                self.num_wins[_pos] = -1
        if win_player_cnt == 1:
            for _pos in ['A', 'B', 'C', 'D']:
                if self.num_wins[_pos] == 1:
                    self.num_wins[_pos] = 3
                    break
        elif win_player_cnt == 3:
            for _pos in ['A', 'B', 'C', 'D']:
                if self.num_wins[_pos] == -1:
                    self.num_wins[_pos] = -3
                    break

    def update_num_wins_scores(self):
        # cal the score
        source_scores = {
            'A': 0,
            'B': 0,
            'C': 0,
            'D': 0
        }
        all_score = 0
        for pos in ['A', 'B', 'C', 'D']:
            # all score poker
            if contain(self.info_sets[pos].won_cards[pos], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 48, 34]):
                # source_scores[pos] = 800
                source_scores[pos] = 8
                if 12 in self.all_showed_card:
                    source_scores[pos] = source_scores[pos] + 2
                if 23 in self.all_showed_card:
                    source_scores[pos] = source_scores[pos] + 1
                if 48 in self.all_showed_card:
                    source_scores[pos] = source_scores[pos] + 1
                if 34 in self.all_showed_card:
                    source_scores[pos] = source_scores[pos] * 2
                continue
            # all red poker
            if contain(self.info_sets[pos].won_cards[pos], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
                # source_scores[pos] = 200
                source_scores[pos] = 2
                if 12 in self.all_showed_card:
                    source_scores[pos] = 4
                for _card in self.info_sets[pos].won_cards[pos]:
                    if _card[0] in score_card.keys() and _card[0] not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                        if _card[0] in self.all_showed_card:
                            source_scores[pos] += score_card.get(_card[0]) * 2
                        else:
                            source_scores[pos] += score_card.get(_card[0])
                if [34] in self.info_sets[pos].won_cards[pos]:
                    source_scores[pos] = source_scores[pos] * 2
                    if 34 in self.all_showed_card:
                        source_scores[pos] = source_scores[pos] * 2
                continue
            flag = True
            for _card in self.info_sets[pos].won_cards[pos]:
                if _card[0] in score_card.keys():
                    flag = False
                    break
            # just have M10
            if flag and 34 in self.info_sets[pos].won_cards[pos]:
                if 34 in self.all_showed_card:
                    source_scores[pos] = 1
                else:
                    source_scores[pos] = 0.5
                continue
            score = 0
            double = 1
            if [34] in self.info_sets[pos].won_cards[pos]:
                double = 2
                if 34 in self.all_showed_card:
                    double = 4
            for _card in self.info_sets[pos].won_cards[pos]:
                if _card[0] in score_card.keys():
                    if _card[0] in self.all_showed_card:
                        score += score_card[_card[0]] * double * 2
                    else:
                        score += score_card[_card[0]] * double
            source_scores[pos] = score
            all_score += source_scores[pos]
        for pos in ['A', 'B', 'C', 'D']:
            self.num_scores[pos] = 3 * source_scores[pos] - (all_score - source_scores[pos])

    def step(self, write_file=False, game_log_file_path=""):
        action = self.players[self.acting_player_position].act(
            self.game_infoset)
        assert action in self.game_infoset.legal_actions
        if write_file:
            with open(game_log_file_path, "a") as f:
                f.write("P " + self.acting_player_position + " " + str(action[0]) + "\n")
        self.loop_move[self.acting_player_position] = action.copy()
        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)

        self.played_cards[self.acting_player_position] += action
        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()

    def get_one_loop_first_move(self):
        if len(self.card_play_action_seq) % 4 == 0:
            return []
        if len(self.card_play_action_seq) % 4 == 1:
            return self.card_play_action_seq[-1]
        if len(self.card_play_action_seq) % 4 == 2:
            return self.card_play_action_seq[-2]
        if len(self.card_play_action_seq) % 4 == 3:
            return self.card_play_action_seq[-3]

    def get_one_loop_second_moves(self):
        if len(self.card_play_action_seq) % 4 == 2:
            return self.card_play_action_seq[-1]
        if len(self.card_play_action_seq) % 4 == 3:
            return self.card_play_action_seq[-2]
        return []

    def get_one_loop_third_moves(self):
        if len(self.card_play_action_seq) % 4 == 3:
            return self.card_play_action_seq[-1]
        return []

    def get_acting_player_position(self):
        # the action player position
        if self.acting_player_position is None:
            self.acting_player_position = 'A'
        else:
            # 如果一轮打完 判断下一轮是谁
            if len(self.card_play_action_seq) % 4 == 0:
                self.loop_move = {'A': [],
                                  'B': [],
                                  'C': [],
                                  'D': []}
                card_type = int(self.card_play_action_seq[-4][0] / 13)
                self.first_play[card_type] = True
                history_cards = self.card_play_action_seq[-4:]
                max_card = self.card_play_action_seq[-4][0]
                max_card_player = 0
                cnt = 0
                for _card in history_cards:
                    # the same type
                    if _card[0] >= card_type * 13 and _card[0] <= (card_type + 1) * 13 - 1:
                        if _card[0] > max_card:
                            max_card = _card[0]
                            max_card_player = cnt
                    cnt += 1
                player_sort = 3 - max_card_player
                player_list = ['A', 'B', 'C', 'D']
                win_player_index = player_list.index(self.acting_player_position)
                for i in range(player_sort):
                    win_player_index -= 1
                    if win_player_index < 0:
                        win_player_index = 3
                win_player = player_list[win_player_index]
                # self.info_sets[win_player].win_cards.extend(history_cards)
                self.won_cards[win_player].extend(history_cards)
                self.acting_player_position = win_player
                return win_player
            elif self.acting_player_position == 'A':
                self.acting_player_position = 'B'

            elif self.acting_player_position == 'B':
                self.acting_player_position = 'C'

            elif self.acting_player_position == 'C':
                self.acting_player_position = 'D'
            elif self.acting_player_position == 'D':
                self.acting_player_position = 'A'

        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        action = action[0]
        self.info_sets[
            self.acting_player_position].player_hand_cards.remove(action)
        self.info_sets[self.acting_player_position].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        hand_cards = self.info_sets[self.acting_player_position].player_hand_cards
        action_sequence = self.card_play_action_seq
        move = []
        card_type = [[], [], [], []]
        showed_card = [12, 23, 34, 48]
        if len(action_sequence) % 4 == 0:
            for card in hand_cards:
                card_type[int(card / 13)].append(card)
                move.append([card])
            # 红黑梅方
            for _type in range(4):
                if not self.first_play[_type] and showed_card[_type] in hand_cards \
                        and showed_card[_type] in self.all_showed_card and len(card_type[_type]) > 1:
                    move.remove([showed_card[_type]])
            return move
        else:
            # 获取首轮出牌花色
            seq_len = len(action_sequence)
            type_index = int(seq_len / 4)
            _type = int(action_sequence[type_index * 4][0] / 13)
            for card in hand_cards:
                hand_type = int(card / 13)
                if hand_type == _type:
                    move.append([card])
                card_type[hand_type].append(card)
            if len(move) == 0:
                # 垫牌 都可以出 没有show card限制
                for i in hand_cards:
                    move.append([i])
            else:
                # 删掉show card
                if not self.first_play[_type] and showed_card[_type] in hand_cards \
                        and showed_card[_type] in self.all_showed_card and len(card_type[_type]) > 1:
                    move.remove([showed_card[_type]])
            return move

    def reset(self):
        self.card_play_action_seq = []

        self.game_over = False

        self.acting_player_position = None

        # the card type first loop and whether show card can be play
        self.first_play = [False, False, False, False]

        # infoset中的引用
        self.played_cards = {'A': [],
                             'B': [],
                             'C': [],
                             'D': []}

        # infoset中的引用
        self.won_cards = {'A': [],
                          'B': [],
                          'C': [],
                          'D': []}

        self.showed_card = {'A': [],
                            'B': [],
                            'C': [],
                            'D': []}

        self.loop_move = {'A': [],
                          'B': [],
                          'C': [],
                          'D': []}

        self.all_showed_card = []

        self.num_wins = {'A': 0,
                         'B': 0,
                         'C': 0,
                         'D': 0}

        self.num_scores = {'A': 0,
                           'B': 0,
                           'C': 0,
                           'D': 0}

        self.info_sets = {'A': InfoSet('A'),
                          'B': InfoSet('B'),
                          'C': InfoSet('C'),
                          'D': InfoSet('D')}

        for _pos in ['A', 'B', 'C', 'D']:
            self.info_sets[_pos].won_cards = self.won_cards
            self.info_sets[_pos].showed_cards = self.showed_card
            self.info_sets[_pos].played_cards = self.played_cards
            self.info_sets[_pos].loop_move = self.loop_move

    def get_infoset(self):
        self.info_sets[
            self.acting_player_position].legal_actions = \
            self.get_legal_card_play_actions()

        self.info_sets[self.acting_player_position].other_hand_cards = []

        # 除了自己的牌和已经出的牌之外的牌
        for pos in ['A', 'B', 'C', 'D']:
            if pos != self.acting_player_position:
                self.info_sets[
                    self.acting_player_position].other_hand_cards += \
                    self.info_sets[pos].player_hand_cards

        self.info_sets[self.acting_player_position].played_cards = \
            self.played_cards

        self.info_sets[self.acting_player_position].loop_move = self.loop_move

        self.info_sets[self.acting_player_position].card_play_action_seq = \
            self.card_play_action_seq
        self.info_sets[self.acting_player_position].won_cards = self.won_cards

        self.info_sets[self.acting_player_position].showed_cards = self.showed_card

        # self.info_sets[self.acting_player_position].loop_move = [self.get_one_loop_first_move(),
        #                                                          self.get_one_loop_second_moves(),
        #                                                          self.get_one_loop_third_moves()]

        return deepcopy(self.info_sets[self.acting_player_position])

    def update_show_card(self, position, card_array):
        self.showed_card[position].extend(card_array)
        self.all_showed_card.extend(card_array)


class InfoSet(object):

    def __init__(self, player_position):
        # The player position
        self.player_position = player_position
        # The player current hand cards. list
        self.player_hand_cards = None
        # The current other hand cards. list
        self.other_hand_cards = None
        # The loop move. ['A', 'B', 'C', 'D'] -> dict()
        self.loop_move = None
        # The played cards. ['A', 'B', 'C', 'D'] -> dict()
        self.played_cards = None
        # The won cards. ['A', 'B', 'C', 'D'] -> dict()
        self.won_cards = None
        # The showed cards. ['A', 'B', 'C', 'D'] -> dict()
        self.showed_cards = None
        # The card play seq
        self.card_play_action_seq = None
        # The legal action
        self.legal_actions = None

    def string(self):
        print("player position: " + str(self.player_position))
        print("player hand cards: " + str(self.player_hand_cards))
        print("legal action: " + str(self.legal_actions))
        print("other hand cards: " + str(self.other_hand_cards))
        print("loop move: " + str(self.loop_move))
        print("played cards: " + str(self.played_cards))
        print("won cards: " + str(self.won_cards))
        print("showed cards: " + str(self.showed_cards))
