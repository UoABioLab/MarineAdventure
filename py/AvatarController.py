# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:03:17 2024

@author: jyr97
"""
import pygame
import time
import initialization as gi


class Avatar(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_frames = [pygame.image.load(f'avatar_frame_{i}.png').convert_alpha() for i in range(1, 3)]
        self.current_frame = 0
        self.frames = self.scale_frames(self.original_frames, gi.SCREEN_WIDTH, gi.SCREEN_HEIGHT)
        self.image = self.frames[self.current_frame]
        self.rect = self.image.get_rect()
        self.rect.center = (gi.SCREEN_WIDTH*0.3, gi.SCREEN_HEIGHT // 2)
        self.animation_speed = 0.1
        self.last_update_time = time.time()

    def scale_frames(self, frames, width, height):
        scale_factor = min(width / 800, height / 600)  # 根据窗口大小的比例
        frame_size = (int(80 * scale_factor), int(80 * scale_factor))  # 调整后的大小
        return [pygame.transform.scale(frame, frame_size) for frame in frames]

    def update_position(self, y_pos):
        self.rect.y = y_pos

    def update(self):
        current_time = time.time()
        if current_time - self.last_update_time > self.animation_speed:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.image = self.frames[self.current_frame]
            self.last_update_time = current_time

    def resize(self, width, height):
        self.frames = self.scale_frames(self.original_frames, width, height)
        self.image = self.frames[self.current_frame]
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect = self.image.get_rect(center=(width * 0.3, self.rect.center[1]))