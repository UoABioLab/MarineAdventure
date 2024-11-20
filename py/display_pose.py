import cv2
import numpy as np
import pygame
from mediapipe import solutions as mp_solutions
import color

def display_pose(frame, results, screen):
    # 创建一个较大的显示区域
    display_width = int(screen.get_width() * 0.25)  # 屏幕宽度的25%
    display_height = int(screen.get_height() * 0.4)  # 屏幕高度的40%
    
    # 调整frame大小以适应显示区域
    frame = cv2.resize(frame, (display_width, display_height))
    
    if results and results.pose_landmarks:
        # 在调整大小后的frame上绘制骨架
        temp_frame = frame.copy()
        
        # 绘制关键点和连接线
        mp_solutions.drawing_utils.draw_landmarks(
            temp_frame, 
            results.pose_landmarks, 
            mp_solutions.pose.POSE_CONNECTIONS,
            mp_solutions.drawing_styles.get_default_pose_landmarks_style())
        
        frame = temp_frame

    try:
        # 转换为Pygame可以使用的格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转换为可供 pygame 使用的格式
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
    except Exception as e:
        print(f"转换图像格式时出错: {e}")
        return

    try:
        # 修改位置：左下角
        margin = 20  # 边距
        
        # 计算左下角的位置
        x_pos = margin
        y_pos = screen.get_height() - display_height - margin
        
        # 创建一个背景矩形
        background_rect = pygame.Rect(
            x_pos - 10,  # 稍微向左偏移以容纳边框
            y_pos - 10,  # 稍微向上偏移以容纳边框
            display_width + 20,
            display_height + 20
        )
        
        # 绘制背景和边框
        pygame.draw.rect(screen, color.DARKGRAY, background_rect)  # 背景
        pygame.draw.rect(screen, color.WHITE, background_rect, 2)  # 边框

        # 显示摄像头画面
        screen.blit(frame_surface, (x_pos, y_pos))

        # 添加标题
        font = pygame.font.SysFont("Comic Sans MS", 24)
        title = font.render("Camera View", True, color.WHITE)
        title_rect = title.get_rect(center=(
            x_pos + display_width//2,
            y_pos - 25
        ))
        screen.blit(title, title_rect)

        # 如果检测到姿势，显示提示信息
        if results and results.pose_landmarks:
            status_text = "Pose Detected"
            status_color = color.GREEN
        else:
            status_text = "No Pose Detected"
            status_color = color.RED
        
        status = font.render(status_text, True, status_color)
        status_rect = status.get_rect(center=(
            x_pos + display_width//2,
            y_pos + display_height + 15
        ))
        screen.blit(status, status_rect)
        
    except Exception as e:
        print(f"显示摄像头画面时出错: {e}")
