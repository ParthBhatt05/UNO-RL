import sys
import time
import pygame
import numpy as np
from keras.models import load_model
from environment import UnoEnvironment
from renderer import *

POSSIBLE_PLAYER_TYPES = ['DQN', 'Manual', 'Random']

player_types = []
player_names = []
for player in sys.argv[1:]:
    found = False
    for i, possible in enumerate(POSSIBLE_PLAYER_TYPES):
        if player.lower() == possible.lower():
            player_types.append(i)
            player_names.append(f'{possible}-{np.sum([i == player_type for player_type in player_types])}')
            found = True
            break

pygame.init()
screen = pygame.display.set_mode((1920, 1080))
clock = pygame.time.Clock()
font_large = pygame.font.SysFont('Times New Roman', 20, bold=True)
font_small = pygame.font.SysFont('Times New Roman', 14)

if 0 in player_types:
    model = load_model('models/model-10000.h5')

prompts = []
previous_move = time.time()

env = UnoEnvironment(len(player_types))

mouse_down = False
clicked = False
done = False
finish = False

while not done:
    clicked = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                clicked = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = False

    if not finish:
        screen.fill((180, 180, 180))
        _cards = draw_env(env, screen, font_large, player_names, player_types)
        draw_messages(prompts, screen, font_small)
        pygame.display.flip()

        if player_types[env.turn] == 0:
            if time.time() - previous_move < 0:
                action = None
            else:
                state = env.get_possible_moves()
                action = np.argmax(model.predict(state.reshape((1, -1)))[0])

                if not env.is_valid(action):
                    prompts.append((time.time(), f'{player_names[env.turn]} selected an illegal action, playing a random card.'))
                    while not env.is_valid(action):
                        action = np.random.randint(env.possible_actions())
        elif player_types[env.turn] == 1:
            card_selected = False
            if clicked:
                mouse_pos = pygame.mouse.get_pos()
                pos = np.argwhere([rect.contains(mouse_pos + (0, 0)) for rect in _cards])
                if len(pos) > 0:
                    if pos[0,0] == np.sum(env.num_players[env.turn].cards):
                        action = len(UnoEnvironment.ALL_CARDS)
                    else:
                        cards = [[pos] * int(count) for pos, count in enumerate(env.num_players[env.turn].cards) if count > 0]
                        cards = np.concatenate(cards)
                        action = cards[pos[0,0]]

                    if env.is_valid(action):
                        card_selected = True
                    else:
                        prompts.append((time.time(), 'Illegal move!'))
            if not card_selected:
                action = None
        elif player_types[env.turn] == 2:
            if time.time() - previous_move < 0:
                action = None
            else:
                action = 0
                while not env.is_valid(action):
                    action += 1

        if action is not None:
            _, _, finish, info = env.make_move(action)
            previous_move = time.time()

            turn = info['turn']
            player_status = info['player']

            if player_status == -1 or player_status == 2:
                if player_status == -1:
                    prompts.append((time.time(), f'{player_names[turn]} is eliminated due to illegal move.'))
                elif player_status == 2:
                    prompts.append((time.time(), f'{player_names[turn]} has finished!'))
                del player_types[turn]
                del player_names[turn]

            if finish:
                screen.fill((180, 180, 180))
                draw_env(env, screen, font_large, player_names, player_types)
                draw_messages(prompts, screen, font_small)
                pygame.display.flip()

    prompts = [msg for msg in prompts if time.time() - msg[0] < 3]

    clock.tick(1)

pygame.quit()
