import random

class RandomAgent():

    def __init__(self):
        self.name = 'Random'

    def act(self, infoset):
        # infoset.string()
        return random.choice(infoset.legal_actions)
