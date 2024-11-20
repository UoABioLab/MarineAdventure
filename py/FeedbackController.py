# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:11:03 2024

@author: jyr97
"""

import pygame
import time
import color
#import initialization as gi

class FeedbackController:
    def __init__(self):
        # 初始化声音
        self.collision_sound = pygame.mixer.Sound('collision.wav')  # 确保collision.wav文件存在
        self.collision_sound.set_volume(1)
        self.display_duration = 1  # 显示 "-1" 的持续时间（秒）
        self.display_start_time = None

    def play_collision_sound(self):
        self.collision_sound.play()

    def start_display_score_decrease(self):
        self.display_start_time = time.time()

    def display_score_decrease(self, screen, font, delta):
        if self.display_start_time is not None:
            elapsed_time = time.time() - self.display_start_time
            if elapsed_time < self.display_duration:
                # 显示 "-1" 的效果
                
                text = font.render(delta, True, (255, 0, 0))
                text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
                screen.blit(text, text_rect)
            else:
                self.display_start_time = None  # 重置显示时间
                
class StarController:
    def __init__(self):
        self.collision_sound = pygame.mixer.Sound('coin.mp3') 
        self.collision_sound.set_volume(1)
        self.stars = 0
        self.font = pygame.font.SysFont("Comic Sans MS", 64)
        self.display_start_time = None
        self.display_duration = 1  # 显示 "-1" 的持续时间（秒）
    
    def play_collision_sound(self):
        self.collision_sound.play()
    
    def start_display_score_increase(self):
        self.display_start_time = time.time()

    def display_score_increase(self, screen, font, delta):
        if self.display_start_time is not None:
            elapsed_time = time.time() - self.display_start_time
            if elapsed_time < self.display_duration:
                # 显示 "+1" 的效果
                
                text = font.render(delta, True, color.GREEN)
                text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
                screen.blit(text, text_rect)
            else:
                self.display_start_time = None  # 重置显示时间

        
    




