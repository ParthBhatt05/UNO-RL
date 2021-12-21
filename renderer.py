import pygame
import numpy as np
from environment import UnoEnvironment

CARD_COLORS = [(255, 0, 0), (0, 255, 0), (0, 150, 255), (255, 255, 0), (75, 75, 75)]

def draw_card(pos, card, surface, font, selected=False, flipped=False):
    if type(card) == str:
        color_index, card_type = 4, -1
    elif hasattr(card, '__iter__'):
        color_index, card_type = card
    else:
        color_index, card_type = UnoEnvironment.CARD_TYPES[card]

    color = CARD_COLORS[4]
    if color_index is not None:
        color = CARD_COLORS[color_index]

    if card_type == -1:
        card_text = '+'
    elif card_type < 10:
        card_text = str(card_type)
    elif card_type == 10:
        card_text = '<-'
    elif card_type == 11:
        card_text = '+2'
    elif card_type == 12:
        card_text = 'O'
    elif card_type == 13:
        card_text = 'W'
    elif card_type == 14:
        card_text = '+4'

    rect = pygame.rect.Rect((pos[0] - 40 // 2, pos[1] - 60 // 2, 40, 60))

    if selected and rect.contains(pygame.mouse.get_pos() + (0, 0)):
        rect.x += 2
        rect.y += 2
        rect.width -= 2 * 2
        rect.height -= 2 * 2

    if flipped:
        color = CARD_COLORS[-1]
    surface.fill(color, rect)

    if not flipped:
        text = font.render(card_text, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = pos
        surface.blit(text, text_rect)

def draw_player(cards, has_turn, offset, surface, font, is_human):
    if has_turn:
        x = offset[0] - 40 / 2 - 5
        y = offset[1] - (60 + 5) / 2
        width = np.sum(cards) * (40 + 5) + 5
        height = 60 + 5
        surface.fill((255, 255, 255), (x, y, width, height))

    player_rects = []
    x, y = offset
    for card_index in np.argwhere(cards > 0)[:,0]:
        for _ in range(cards[card_index].astype(int)):
            draw_card((x, y), card_index, surface, font, selected=has_turn and is_human, flipped=not is_human)
            player_rects.append(pygame.rect.Rect((x - 40 / 2, y - 60 / 2, 40, 60)))
            x += 40 + 5
    return player_rects

def draw_env(env, surface, font, names, types, draw_non_human=True):
    x = 40 / 2 + 5
    y = surface.get_bounding_rect().center[1] - (60 + 5) / 2
    draw_card((x, y), env.top_card, surface, font)
    y = surface.get_bounding_rect().center[1] + (60 + 5) / 2
    draw_card((x, y), 'stack', surface, font, selected=types[env.turn] == 1)
    stack_rect = pygame.rect.Rect((x - 40 / 2, y - 60 / 2, 40, 60))

    card_rects = None
    for i, (player, name, type) in enumerate(zip(env.players, names, types)):
        x_offset = 20 + 40 * 1.5
        y_offset = 5 + i * (60 + 5) + 60 / 2
        text = font.render(name, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.midleft = (x_offset, y_offset)
        surface.blit(text, text_rect)
        x_offset += 100
        rects = draw_player(player.cards, i == env.turn, (x_offset, y_offset), surface, font, type == 1)

        if i == env.turn:
            card_rects = rects

    card_rects.append(stack_rect)
    return card_rects

def draw_messages(messages, surface, font):
    message_list = [msg[1] for msg in messages]
    screen_rect = surface.get_bounding_rect()
    for i, msg in enumerate(message_list):
        text = font.render(msg, True, (0, 0, 0))
        text_rect = text.get_rect()
        x = screen_rect.width - 5
        y = screen_rect.height - (i + 1) * (font.get_height())
        text_rect.midright = (x, y)
        surface.blit(text, text_rect)
