# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:46:53 2024

@author: jyr97
"""

def set_difficulty_parameters(difficulty):
    if difficulty == "Easy":
        return {
            'OBSTACLE_MIN_GAP': 11,
            'OBSTACLE_MAX_GAP': 13,
            'TOP_OBSTACLE_MIN_HEIGHT': 30,
            'BOTTOM_OBSTACLE_MIN_HEIGHT': 30,
            'height_index_up': 5,
            'height_index_down': 2,
            'BONUS_ITEM_INTERVAL': 30,
            'top_count_per_time_low':1,
            'top_count_per_time_high':2,
            'bottom_count_per_time_low':1,
            'bottom_count_per_time_high':2,
            'game_duration': 90
        }
    elif difficulty == "Medium":
        return {
            'OBSTACLE_MIN_GAP': 8,
            'OBSTACLE_MAX_GAP': 10,
            'TOP_OBSTACLE_MIN_HEIGHT': 30,
            'BOTTOM_OBSTACLE_MIN_HEIGHT': 30,
            'height_index_up': 3,
            'height_index_down': 3.5,
            'BONUS_ITEM_INTERVAL': 20,
            'top_count_per_time_low':1,
            'top_count_per_time_high':3,
            'bottom_count_per_time_low':1,
            'bottom_count_per_time_high':2,
            'game_duration': 120
        }
    elif difficulty == "Hard":
        return {
            'OBSTACLE_MIN_GAP': 4,
            'OBSTACLE_MAX_GAP': 6,
            'TOP_OBSTACLE_MIN_HEIGHT': 30,
            'BOTTOM_OBSTACLE_MIN_HEIGHT': 30,
            'height_index_up': 0.5,
            'height_index_down': 5,
            'BONUS_ITEM_INTERVAL': 10,
            'top_count_per_time_low':1,
            'top_count_per_time_high':3,
            'bottom_count_per_time_low':1,
            'bottom_count_per_time_high':2,
            'game_duration': 180
        }
    else:
        raise ValueError("Unknown difficulty level")