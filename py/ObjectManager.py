# game_elements.py
import pygame
import random
import color
import initialization as gi

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, height, position, screen_height):
        super().__init__()
        # self.original_image = pygame.image.load(random.choice(gi.obstacle_images)).convert_alpha()
        # self.image = pygame.transform.scale(self.original_image, (50, height))
        self.image = pygame.Surface((50, height))
        self.image.fill(color.RED)
        self.rect = self.image.get_rect()
        self.rect.x = x
        if position == 'top':
            self.rect.y = 0
        else:
            self.rect.y = screen_height - height

    def update(self):
        self.rect.x -= 5
        if self.rect.x < -50:
            self.kill()

class BonusItem(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.image.load(random.choice(gi.bonus_images)).convert_alpha()
        self.image = pygame.transform.scale(self.original_image, (70, 70))  # Scale the image to the desired size
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def update(self):
        self.rect.x -= 5
        if self.rect.x < -self.rect.width:
            self.kill()
            
    def destroy(self):
        self.kill() 

def add_obstacles(screen_height, height_index_1, height_index_2):
    current_time = pygame.time.get_ticks()
    if gi.calibrated_height is None:
        return

    if current_time - gi.last_obstacle_time > random.randint(gi.OBSTACLE_MIN_GAP * 1000, gi.OBSTACLE_MAX_GAP * 1000):
        if gi.obstacle_toggle == 'top':
            top_obstacle_count = random.randint(
                int(gi.top_count_per_time_low), 
                int(gi.top_count_per_time_high)
            )
            for _ in range(top_obstacle_count):
                min_height = max(30, int(gi.calibrated_height - height_index_1 * gi.TOP_OBSTACLE_MIN_HEIGHT))
                max_height = min(
                    int(screen_height - 50), 
                    int(gi.calibrated_height + gi.TOP_OBSTACLE_MIN_HEIGHT * height_index_2)
                )
                
                if min_height >= max_height:
                    min_height = max_height - 50
                
                height = random.randint(min_height, max_height)
                obstacle = Obstacle(gi.SCREEN_WIDTH + _ * 200, height, 'top', screen_height)
                gi.all_sprites.add(obstacle)
                gi.obstacles.add(obstacle)
            gi.obstacle_toggle = 'bottom'
        else:
            bottom_obstacle_count = random.randint(
                int(gi.bottom_count_per_time_low), 
                int(gi.bottom_count_per_time_high * 0.6)
            )
            for _ in range(bottom_obstacle_count):
                min_height = max(30, int(gi.BOTTOM_OBSTACLE_MIN_HEIGHT))
                max_height = max(50, int(screen_height - gi.calibrated_height - 50))
                
                if min_height >= max_height:
                    max_height = min_height + 50
                
                height = random.randint(min_height, max_height)
                obstacle = Obstacle(gi.SCREEN_WIDTH + _ * 200, height, 'bottom', screen_height)
                gi.all_sprites.add(obstacle)
                gi.obstacles.add(obstacle)
            gi.obstacle_toggle = 'top'
        gi.last_obstacle_time = current_time

def add_bonus_item(screen_height):
    current_time = pygame.time.get_ticks()
    if current_time - gi.last_bonus_time < gi.BONUS_ITEM_INTERVAL * 1000:
        return

    if gi.calibrated_height is None:
        return

    # Generate a random y position within the range
    bonus_y_range = list(range(max(gi.calibrated_height - 50, 50), min(gi.calibrated_height + 200, gi.SCREEN_HEIGHT - 50)))
    valid_y_positions = []

    # Check for valid y positions that do not overlap with obstacles
    buffer_distance = 150  # Minimum distance to keep from obstacles
    for y in bonus_y_range:
        potential_bonus_rect = pygame.Rect(gi.SCREEN_WIDTH, y, 30, 30)
        collision = False
        for obstacle in gi.obstacles:
            if potential_bonus_rect.colliderect(obstacle.rect.inflate(buffer_distance, buffer_distance)):
                collision = True
                break
        if not collision:
            valid_y_positions.append(y)

    if valid_y_positions:
        bonus_y = random.choice(valid_y_positions)
        bonus_item = BonusItem(gi.SCREEN_WIDTH, bonus_y)
        gi.all_sprites.add(bonus_item)
        gi.bonus_items.add(bonus_item)
        gi.last_bonus_time = current_time
