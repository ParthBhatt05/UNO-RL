import numpy as np

class UnoEnvironment:

    ALL_CARDS = [[color, type] for color in range(4) for type in range(13)]
    ALL_CARDS += [[None, 13], [None, 14]]
    ALL_CARDS = np.array(ALL_CARDS)

    def __init__(self, num_players):
        self.num_players = num_players
        self.reset()
    
    def _next_turn(self):
        self.turn += self.normal_reversed

        if self.turn < 0:
            self.turn = len(self.players) - 1
        elif self.turn >= len(self.players):
            self.turn = 0
    
    def is_valid(self, action, player=None):
        
        if player is None:
            player = self.players[self.turn]

        if action == len(self.ALL_CARDS):
            return True
        
        card = self.ALL_CARDS[action]

        if player.cards[action] == 0:
            return False
        if self.to_draw > 0 and self.card_stack[1] == 11 and card[1] != 11:
            return False
        if self.to_draw > 0 and self.card_stack[1] == 14 and card[1] != 14:
            return False
        if card[1] == 13 or card[1] == 14:
            return True

        return self.card_stack[0] == card[0] or self.card_stack[1] == card[1]

    def possible_actions(self):
        return len(self.ALL_CARDS) + 1

    def get_num_players(self):
        return len(self.players)

    def get_possible_moves(self):
        
        colored_cards = np.all(self.ALL_CARDS[:-2] == self.card_stack, axis=1).astype(np.float)

        wild = np.zeros(4)
        if self.card_stack[1] == 13:
            wild[self.card_stack[0]] = 1

        draw_4 = np.zeros(4)
        if self.card_stack[1] == 14:
            draw_4[self.card_stack[0]] = 1

        cards_player = self.players[self.turn].cards

        return np.concatenate([colored_cards, wild, draw_4, cards_player, [self.to_draw]])
    
    def make_move(self, action):
        reward = 0
        i = self.turn
        status = None
        player = self.players[self.turn]
        
        card_played = None
        if action < len(self.ALL_CARDS):
            card_played = self.ALL_CARDS[action].copy()

        if self.is_valid(action):
            if self.to_draw > 0:
                if card_played is None:
                    player.draw_cards(self.to_draw)
                    self.to_draw = 0
                    status = 0
                elif self.card_stack[1] == 11 and card_played[1] == 11:
                    self.to_draw += 2
                    status = 1
                elif self.card_stack[1] == 14 and card_played[1] == 14:
                    self.to_draw += 4
                    status = 1
            elif card_played is None:
                player.draw_cards(1)
                status = 0
            elif card_played[1] == 10:
                self.normal_reversed *= -1
                status = 1
            elif card_played[1] == 11:
                self.to_draw = 2
                status = 1
            elif card_played[1] == 12:
                self._next_turn()
                status = 1
            elif card_played[1] == 13:
                card_played[0] = np.random.randint(4)
                status = 1
            elif card_played[1] == 14:
                self.to_draw = 4
                card_played[0] = np.random.randint(4)
                status = 1
            else:
                status = 1
        else:
            status = -1

        if status == -1:
            reward -= -2
            self._eliminate(player)
        elif status == 0:
            reward -= 1
        elif status == 1:
            reward += 2
            player.play_card(action, card_played[0])

        if player.num_cards() == 0:
            status = 2
            reward += 10
            self._eliminate(player)

        if card_played is not None:
            self.card_stack = card_played

        
        done = len(self.players) <= 1
        self._next_turn()

        return self.get_possible_moves(), reward, done, {'turn': i, 'player': status}
        
    def reset(self):
        self.players = [UnoPlayer(self, num_cards=7) for _ in range(self.num_players)]
        self.card_stack = self.ALL_CARDS[np.random.randint(len(self.ALL_CARDS))]
        if self.card_stack[0] is None:
            self.card_stack[0] = np.random.randint(0, 4)
        self.to_draw = 0
        self.turn = 0
        self.normal_reversed = 1

    def _eliminate(self, player):
        self.players.remove(player)
        if self.normal_reversed == 1:
            self.turn -= 1

    def num_possible_moves(self):
        return len(self.get_possible_moves())


class UnoPlayer:

    def __init__(self, game, num_cards=7):
        self.game = game
        self.cards = np.zeros(len(self.game.ALL_CARDS), dtype=np.float)
        self.draw_cards(num_cards)

    def draw_cards(self, count):
        for _ in range(count):
            self.cards[np.random.randint(len(self.game.ALL_CARDS))] += 1

    def play_card(self, card_index, color=None):
        if not self.game.is_valid(card_index, player=self):
            return False
        self.cards[card_index] -= 1
        card = self.game.ALL_CARDS[card_index].copy()
        if color is not None:
            card[0] = color
        return card

    def num_cards(self):
        return int(sum(self.cards))

    def __repr__(self):
        return f'Player({self.num_cards()})'
