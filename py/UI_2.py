import pygame
import color


class Button:
    def __init__(self, x, y, w, h, text, color, hover_color, font_size):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_visible = True
        self.font_size = font_size
        self.font = pygame.font.Font(None, self.font_size)
        self.is_enabled = True
        

    def update_position(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = pygame.font.Font(None, self.font_size)

    def draw(self, screen):
        if not self.is_visible:
            return
        else:
            mouse_pos = pygame.mouse.get_pos()
            current_color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
            pygame.draw.rect(screen, current_color, self.rect)
            text_surf = self.font.render(self.text, True, (255, 255, 255))
            screen.blit(text_surf, (self.rect.x + (self.rect.width - text_surf.get_width()) // 2,
                                    self.rect.y + (self.rect.height - text_surf.get_height()) // 2))

    def is_clicked(self, event):
        # if not self.is_visible:
        #     self.is_enabled = False
            
        # else:
        #     if self.is_enabled and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 左键点击
        #         return self.rect.collidepoint(event.pos)
            
        # if self.is_visible:
        #     if self.is_enabled and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 左键点击
        #         return self.rect.collidepoint(event.pos)
        # else: 
        #     self.is_enabled = False
        if not self.is_visible:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False



class Dropdown:
    def __init__(self, x, y, w, h, font_size, main, options):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color.BLUE
        self.hover_color = color.DARK_GREEN
        self.font_size = font_size
        self.font = pygame.font.Font(None, self.font_size)
        self.main = main
        self.options = options
        self.selected = main
        self.active = False
        self.update_options_rects()

    def update_position(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = pygame.font.Font(None, self.font_size)
        self.update_options_rects()

    def update_options_rects(self):
        self.options_rects = [pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width, self.rect.height) for i in range(len(self.options))]

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surf = self.font.render(self.selected, True, pygame.Color('white'))
        screen.blit(text_surf, (self.rect.x + (self.rect.width - text_surf.get_width()) // 2, self.rect.y + (self.rect.height - text_surf.get_height()) // 2))

        if self.active:
            for i, option in enumerate(self.options):
                option_rect = self.options_rects[i]
                pygame.draw.rect(screen, self.hover_color if option_rect.collidepoint(pygame.mouse.get_pos()) else self.color, option_rect)
                option_surf = self.font.render(option, True, pygame.Color('white'))
                screen.blit(option_surf, (option_rect.x + (option_rect.width - option_surf.get_width()) // 2, option_rect.y + (option_rect.height - option_surf.get_height()) // 2))

    def update(self, event):
        previous_active = self.active
        previous_selected = self.selected
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            elif self.active:
                for i, option_rect in enumerate(self.options_rects):
                    if option_rect.collidepoint(event.pos):
                        self.selected = self.options[i]
                        self.active = False
        
        return self.active != previous_active or self.selected != previous_selected
    
    
def adjust_button_position(screen_width, screen_height, dropdown, start_button, quit_button, back_button=None, input_box=None):
    # 计算新的尺寸和位置
    new_width = int(screen_width * 0.3)  # 宽度为屏幕宽度的30%
    new_height = int(screen_height * 0.08)  # 高度为屏幕高度的8%
    new_x = (screen_width - new_width) // 2  # 居中
    new_dropdown_y = int(screen_height * 0.4)  # 下拉菜单在屏幕40%的位置

    # 更新字体大小
    new_font_size = int(min(new_width * 0.1, new_height * 0.6))  # 根据新尺寸调整字体大小

    # 更新下拉菜单
    dropdown.update_position(new_x, new_dropdown_y, new_width, new_height)
    dropdown.font_size = new_font_size
    dropdown.font = pygame.font.SysFont("Comic Sans MS", dropdown.font_size)

    # 更新 InputBox 的位置和尺寸（放在下拉菜单上方）
    if input_box:
        input_box_y = new_dropdown_y - new_height - int(screen_height * 0.02)
        input_box.update_position(new_x, input_box_y)
        input_box.font = pygame.font.SysFont("Comic Sans MS", new_font_size)

    # 根据下拉菜单的状态调整开始按钮的位置和可见性
    if dropdown.active:
        start_button_y = dropdown.rect.y + dropdown.rect.height * (len(dropdown.options) + 1) + int(screen_height * 0.02)
        start_button.is_visible, start_button.is_enabled, quit_button.is_visible, quit_button.is_enabled = False, False, False, False
    else:
        start_button_y = dropdown.rect.y + dropdown.rect.height + int(screen_height * 0.02)
        start_button.is_visible, start_button.is_enabled, quit_button.is_visible, quit_button.is_enabled = True, True, True, True

    # 更新开始按钮
    start_button.update_position(new_x, start_button_y, new_width, new_height)
    start_button.font_size = new_font_size
    start_button.font = pygame.font.SysFont("Comic Sans MS", start_button.font_size)

    # 更新退出按钮的位置
    quit_button_y = start_button_y + new_height + int(screen_height * 0.02)
    quit_button.update_position(new_x, quit_button_y, new_width, new_height)
    quit_button.font_size = new_font_size
    quit_button.font = pygame.font.SysFont("Comic Sans MS", quit_button.font_size)

    # 如果 back_button 存在，更新它的位置和尺寸
    if back_button:
        back_button_y = int(screen_height * 0.8)  # 按钮在屏幕70%的位置
        back_button.update_position(new_x, back_button_y, new_width, new_height)
        back_button.font_size = new_font_size
        back_button.font = pygame.font.SysFont("Comic Sans MS", back_button.font_size)


class InputBox:
    def __init__(self, x, y, w, h, font, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color('white')
        self.color_active = pygame.Color('white')
        self.color = self.color_inactive
        self.font = font
        self.text = text
        self.default_text = "ID"
        self.txt_surface = self.font.render(self.default_text, True, self.color)
        self.active = False
        self.initial_width = w
        self.initial_height = h
        self.showing_default = True  # 初始化 showing_default 属性
        self.saved_text = ""  # 保存输入内容

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                # 当输入框被点击时，激活输入框
                self.active = True
                if self.showing_default:
                    # 如果正在显示默认文本，清空文本
                    self.text = ''
                    self.showing_default = False
                self.color = self.color_active
            else:
                self.active = False
                self.color = self.color_inactive
                self.saved_text = self.text

            # 如果用户点击输入框外部，且文本框为空，则显示默认文本
            if not self.active and not self.text:
                self.text = self.default_text
                self.showing_default = True
                self.txt_surface = self.font.render(self.text, True, self.color)

        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    # 用户按下Enter键时保存内容
                    self.saved_text = self.text
                    self.active = False  # 取消激活状态
                    self.color = self.color_inactive  # 改变颜色为非激活状态
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, self.color)

    def draw(self, screen):
        text_y = self.rect.y + (self.rect.height - self.txt_surface.get_height()) // 2
        screen.blit(self.txt_surface, (self.rect.x + 5, text_y))
        pygame.draw.rect(screen, self.color, self.rect, 2)

    def update_position(self, x, y):
        self.rect.x = x
        self.rect.y = y
        self.rect.width = self.initial_width
        self.rect.height = self.initial_height

    def get_text(self):
        return self.text if not self.showing_default else ""

    def clear_text(self):
        self.text = ''
        self.txt_surface = self.font.render(self.text, True, self.color)
        self.showing_default = False