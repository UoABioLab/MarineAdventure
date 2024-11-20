# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:59:30 2024

@author: jyr97
"""

import pygame
import time
import color
from AvatarController import Avatar

# 全局变量
avatar = None
all_sprites = None
obstacles = None
score = 5


menu_background_image = None
background_image = None
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = None

start_time = None
obstacle_toggle = 'top'
calibrated_height = None
last_obstacle_time = 0
collision_cooldown = False
calibration_data = []
calibration_phase = True
OBSTACLE_MIN_GAP = 3  # 最小间隔时间（以秒为单位）
OBSTACLE_MAX_GAP = 5  # 最大间隔时间（以秒为单位）
TOP_OBSTACLE_MIN_HEIGHT = 20  # 上方障碍物的最小高度
BOTTOM_OBSTACLE_MIN_HEIGHT = 10  # 下方障碍物的最小高度
height_index_up, height_index_down = 3, 1
top_count_per_time_low,top_count_per_time_high=1,3
bottom_count_per_time_low,bottom_count_per_time_high=1,3
collision_cooldown = True
selected_difficulty = None

last_bonus_time = 0  # 初始化上次奖励生成的时间
BONUS_ITEM_INTERVAL = 5  # 奖励生成的时间间隔，单位：秒

initial_phase = True
ready_phase = True
initial_start_time = None

game_duration = 120




def initialize_game():
    global avatar, all_sprites, obstacles, bonus_items, score, last_obstacle_time, collision_cooldown, height_index_up, height_index_down, background_image, menu_background_image,\
        start_time, obstacle_toggle, calibrated_height, calibration_data, initial_phase, ready_phase, initial_start_time, \
            OBSTACLE_MIN_GAP, OBSTACLE_MAX_GAP, TOP_OBSTACLE_MIN_HEIGHT, BOTTOM_OBSTACLE_MIN_HEIGHT, collision_cooldown, screen,bonuscollision_cooldown, star, bonus_images,obstacle_images
    avatar = Avatar()
    all_sprites = pygame.sprite.Group()
    all_sprites.add(avatar)
    obstacles = pygame.sprite.Group()
    bonus_items = pygame.sprite.Group()
    star = 0
    score = 5  # 初始化分数
    last_obstacle_time = pygame.time.get_ticks()  # 初始化最后一个障碍物的时间
    collision_cooldown = False  # 初始化碰撞冷却状态
    bonuscollision_cooldown = False
    bonus_images = ["bonus1.png", "bonus2.png", "bonus3.png"]
    background_image = pygame.image.load('background.png').convert()  # 加载背景图像
    menu_background_image = pygame.image.load('background_1.jpg').convert()  # 加载菜单背景图像
    start_time = time.time()  # 初始化开始时间
    obstacle_toggle = 'top'
    calibrated_height = None
    calibration_data = []
    initial_phase = True
    ready_phase = True
    initial_start_time = time.time()  # 初始化初始阶段开始时间
    # height_index_up, height_index_down = 3, 1
    # OBSTACLE_MIN_GAP = 3  # 最小间隔时间（以秒为单位）
    # OBSTACLE_MAX_GAP = 5  # 最大间隔时间（以秒为单位）
    # TOP_OBSTACLE_MIN_HEIGHT = 20  # 上方障碍物的最小高度
    # BOTTOM_OBSTACLE_MIN_HEIGHT = 1  # 下方障碍物的最小高度
    collision_cooldown = False
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.mixer.init()
    pygame.mixer.music.load('background music.mp3')
    pygame.mixer.music.set_volume(0.2)
    obstacle_images = ['obstacles_1.png', 'obstacles_1.png']
    
