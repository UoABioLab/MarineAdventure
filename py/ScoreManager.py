# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:05:46 2024

@author: jyr97
"""
import pygame
import initialization as gi
import color
from ObjectManager import BonusItem

def manage_score(feedback_controller):
    if pygame.sprite.spritecollideany(gi.avatar, gi.obstacles):
        if not gi.collision_cooldown:
            gi.score -= 1
            gi.collision_cooldown = True
            feedback_controller.play_collision_sound()  # 播放碰撞声音
            feedback_controller.start_display_score_decrease()  # 显示减分效果
            if gi.score <= 0:
                return True
    else:
        gi.collision_cooldown = False
    return False

def manage_star(star_controller):
    collided_bonus_item = pygame.sprite.spritecollideany(gi.avatar, gi.bonus_items)
    if collided_bonus_item:
        if not gi.bonuscollision_cooldown:
            gi.star += 10
            gi.bonuscollision_cooldown = True
            star_controller.play_collision_sound()  # 播放碰撞声音
            star_controller.start_display_score_increase()  # 显示加分效果
            collided_bonus_item.destroy()  # 销毁碰撞的bonus item实例
    else:
        gi.bonuscollision_cooldown = False
    return False

# LIFT CALCULATION
def draw_score():
    font = pygame.font.SysFont("Comic Sans MS", 52)
    text = font.render(f'Life: {gi.score}', True, color.BLACK)
    gi.screen.blit(text, (gi.SCREEN_WIDTH*0.02, gi.SCREEN_HEIGHT*0.1))

# SCORE CALCULATION 
def draw_star():
    font = pygame.font.SysFont("Comic Sans MS", 52)
    text = font.render(f'Score: {gi.star}', True, color.BLACK)
    gi.screen.blit(text, (gi.SCREEN_WIDTH - 250, gi.SCREEN_HEIGHT*0.1))
