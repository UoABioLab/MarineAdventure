import pygame
import sys
import random
import cv2
import mediapipe as mp
import time
import color

# 导入UIpy中的Button和Dropdown类
from UI_2 import Button, Dropdown, adjust_button_position, InputBox
import initialization as gi
from ScoreManager import manage_score, draw_score, manage_star, draw_star
from FeedbackController import FeedbackController, StarController
from ObjectManager import Obstacle, BonusItem, add_obstacles, add_bonus_item  # 引用新模块中的类和函数
from DifficultyManager import set_difficulty_parameters
from upload_to_goolge_sheets_2 import upload_to_google_sheets
import display_pose as dp

email_to_share = 'chengye171@gmail.com'
# 初始化Pygame
pygame.init()


# 屏幕设置
gi.screen = pygame.display.set_mode((gi.SCREEN_WIDTH, gi.SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption('Shoulder Controlled Game')

# 设置游戏时钟
clock = pygame.time.Clock()
game_end_time = None

# Mediapipe初始化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 全局变量
cap = None
difficulty = None  # 默认难度
back_button = None
landmarks_list = []
landmarks = None


def process_camera_frame():
    """处理摄像头画面"""
    if not cap.isOpened():
        print("摄像头未打开")
        return None, None
        
    try:
        # 清空缓冲区
        cap.grab()
            
        ret, frame = cap.read()
        if not ret or frame is None:
            print("无法读取摄像头画面")
            return None, None

        # 确保图像大小合适
        frame = cv2.resize(frame, (640, 480))
        
        # 水平翻转
        frame = cv2.flip(frame, 1)
        
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用 MediaPipe 处理图像
        results = pose.process(frame_rgb)
        
        return results, frame
        
    except cv2.error as e:
        print(f"OpenCV错误: {e}")
        return None, None
    except Exception as e:
        print(f"处理摄像头画面时出错: {e}")
        return None, None

def capture_all_landmarks(results):
    if results is None:
        return None
        
    try:
        if results.pose_landmarks:
            # 将每个关节点的x, y, z坐标存入一个列表
            landmarks = results.pose_landmarks.landmark     
            landmark_data = []
            for landmark in landmarks:
                landmark_data.extend([landmark.x, landmark.y, landmark.z])
            landmarks_list.append(landmark_data)        
            return landmarks_list
    except Exception as e:
        print(f"处理姿势数据时出错: {e}")
    return None

def get_shoulder_midpoint_y(results):
    if results.pose_landmarks:
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_mid_y = int((shoulder_left.y + shoulder_right.y) / 2 * gi.SCREEN_HEIGHT)
        return shoulder_mid_y
    return None

def draw_timer():
    font = pygame.font.SysFont("Comic Sans MS", 52)
    elapsed_time = time.time() - gi.start_time
    text = font.render(f'Time: {int(elapsed_time)}', True, color.BLACK)
    gi.screen.blit(text, (gi.SCREEN_WIDTH - 250, gi.SCREEN_HEIGHT*0.03))

def draw_initial_countdown():
    global game_end_time
    font = pygame.font.SysFont("Comic Sans MS", 72)
    elapsed_time = time.time() - gi.initial_start_time
    if gi.ready_phase:
        text = font.render("Ready!", True, color.WHITE)
        text_rect = text.get_rect(center=(gi.SCREEN_WIDTH // 2, gi.SCREEN_HEIGHT // 2))
        gi.screen.blit(text, text_rect)
        if elapsed_time > 2:
            gi.ready_phase = False
            gi.initial_start_time = time.time()
    else:
        countdown_time = 5 - int(elapsed_time)
        text = font.render(f"{countdown_time}", True, color.WHITE)
        text_rect = text.get_rect(center=(gi.SCREEN_WIDTH // 2, gi.SCREEN_HEIGHT // 2))
        gi.screen.blit(text, text_rect)
        if countdown_time <= 0:
            gi.initial_phase = False
            gi.calibration_phase = True
            gi.start_time = time.time()
            game_end_time = time.time() + gi.game_duration
            return

    pygame.display.flip()
    
def reset_game():
    """
    重置游戏的所有相关变量和状态，确保新游戏开始时一切归零。
    """
    global cap, game_end_time, final_time,landmarks_list, uploaded
    gi.initialize_game()
    final_time = 0
    landmarks_list = [] 
    uploaded = False  # 重置上传标志
    pygame.mixer.music.play(-1)
    
def draw_and_update_screen(feedback_controller, star_controller, font, font_1, frame=None, results=None):
    # 首先绘制游戏背景
    gi.screen.blit(gi.background_image, [0, 0])
    
    # 绘制游戏元素
    gi.all_sprites.draw(gi.screen)
    
    # 绘制分数和计时器
    draw_star()
    draw_score()
    draw_timer()

    # 显示当前难度
    difficulty_text = font.render(f'Difficulty: {gi.selected_difficulty}', True, color.BLACK)
    gi.screen.blit(difficulty_text, (gi.SCREEN_WIDTH*0.02, gi.SCREEN_HEIGHT*0.03))

    # 显示分数变化效果
    feedback_controller.display_score_decrease(gi.screen, font, '-1')
    star_controller.display_score_increase(gi.screen, font_1, '+10')
    
    # 显示摄像头画面
    if frame is not None and results is not None:
        dp.display_pose(frame, results, gi.screen)
    
    # 最后更新显示
    pygame.display.flip()

def calibrate_height(y_pos):
    gi.calibration_data.append(y_pos)
    if len(gi.calibration_data) >= 120:  # 3秒内收集到的帧数（假设每秒30帧）
        return sum(gi.calibration_data) // len(gi.calibration_data)
    return None

    

def draw_game_over_screen(final_time):
    gi.screen.fill(color.BLACK)
    font = pygame.font.SysFont("Comic Sans MS", 72)
    game_over_text = font.render("Game Over", True, color.WHITE)
    game_over_rect = game_over_text.get_rect(center=(gi.SCREEN_WIDTH // 2, gi.SCREEN_HEIGHT // 3))
    gi.screen.blit(game_over_text, game_over_rect)

    font_small = pygame.font.SysFont("Comic Sans MS", 52)
    time_text = font_small.render(f'Time: {int(final_time)} seconds', True, color.WHITE)
    time_rect = time_text.get_rect(center=(gi.SCREEN_WIDTH // 2, gi.SCREEN_HEIGHT // 2))
    gi.screen.blit(time_text, time_rect)

    score_text = font_small.render(f'Life: {gi.score}', True, color.WHITE)
    score_rect = score_text.get_rect(center=(gi.SCREEN_WIDTH // 2, gi.SCREEN_HEIGHT // 2 + 60))
    gi.screen.blit(score_text, score_rect)

    star_text = font_small.render(f'Score: {gi.star}', True, color.WHITE)
    star_rect = star_text.get_rect(center=(gi.SCREEN_WIDTH // 2, gi.SCREEN_HEIGHT // 2 + 120))
    gi.screen.blit(star_text, star_rect)

    # 绘制返回主菜单的按钮
    if back_button:  # 确保 back_button 已经初始化
        back_button.draw(gi.screen)
    pygame.display.flip()

def list_available_cameras():
    """列出所有可用的摄像头"""
    available_cameras = []
    for i in range(10):  # 检查前10个摄像头索引
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def initialize_camera():
    """初始化摄像头"""
    print("开始检测摄像头...")
    
    try:
        # 首先尝试使用 DSHOW 后端
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        if not cap.isOpened():
            # 如果 DSHOW 失败，尝试不指定后端
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return None
        
        # 设置较低的分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 设置较低的帧率
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 禁用一些高级功能
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 测试是否能读取画面
        for _ in range(5):  # 尝试多次读取
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("摄像头初始化成功")
                print(f"实际分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}")
                return cap
            time.sleep(0.1)  # 短暂延迟
            
        # 如果无法读取画面，释放摄像头
        cap.release()
        return None
            
    except Exception as e:
        print(f"初始化摄像头时出错: {e}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None

def main():
    global cap, difficulty, game_end_time, back_button, final_time  # 声明为全局变量
    reset_game()  # 在游戏开始时重置游戏状态

    uploaded = False  # 用于跟踪是否已经上传过数据

    pygame.mixer.music.play(-1)
    
    # 初始化摄像头
    cap = initialize_camera()
    
    if cap is None:
        print("未能找到可用的摄像头")
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        font = pygame.font.SysFont("Comic Sans MS", 36)
        
        # 显示详细的错误信息
        messages = [
            "无法初始化摄像头，请检查：",
            "1. 确保摄像头已正确连接",
            "2. 尝试重新插拔摄像头",
            "3. 关闭其他使用摄像头的程序",
            "4. 检查摄像头驱动是否正确安装"
        ]
        
        for i, message in enumerate(messages):
            text = font.render(message, True, color.RED)
            screen.blit(text, (50, 150 + i * 50))
        
        pygame.display.flip()
        pygame.time.wait(5000)
        pygame.quit()
        sys.exit()

    feedback_controller = FeedbackController()
    star_controller = StarController()
    running = True
    menu = True
    game_over = False
    font = pygame.font.SysFont("Comic Sans MS", 52)
    font_big = pygame.font.SysFont("Comic Sans MS", 72)
    initial_font_size = 36
    dropdown = Dropdown(300, 200, 250, 60, initial_font_size, "Select Difficulty", ["Easy", "Medium", "Hard"])
    start_button = Button(300, 300, 250, 60, "Start Game", color.BLUE, color.DARK_GREEN, initial_font_size)
    quit_button = Button(300, 400, 250, 60, "Quit", color.BLUE, color.DARKRED, initial_font_size)
    back_button = Button(gi.SCREEN_WIDTH // 2 - 125, gi.SCREEN_HEIGHT // 2 + 180, 250, 60, "Back to Menu", color.BLUE, color.GREEN, initial_font_size)
    input_box = InputBox(gi.SCREEN_WIDTH // 2 - 100, gi.SCREEN_HEIGHT // 2 - 150, 200, 40, font)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                gi.SCREEN_WIDTH, gi.SCREEN_HEIGHT = event.w, event.h
                gi.screen = pygame.display.set_mode((gi.SCREEN_WIDTH, gi.SCREEN_HEIGHT), pygame.RESIZABLE)
                gi.avatar.resize(gi.SCREEN_WIDTH, gi.SCREEN_HEIGHT)

            if menu:
                dropdown.update(event)
                input_box.handle_event(event)

                if dropdown.selected:
                    gi.selected_difficulty = dropdown.selected
                if start_button.is_clicked(event):
                    reset_game()  # 每次开始新游戏时重置状态
                    
                    difficulty = gi.selected_difficulty
                    params = set_difficulty_parameters(difficulty)
                    gi.OBSTACLE_MIN_GAP = params['OBSTACLE_MIN_GAP']
                    gi.OBSTACLE_MAX_GAP = params['OBSTACLE_MAX_GAP']
                    gi.TOP_OBSTACLE_MIN_HEIGHT = params['TOP_OBSTACLE_MIN_HEIGHT']
                    gi.BOTTOM_OBSTACLE_MIN_HEIGHT = params['BOTTOM_OBSTACLE_MIN_HEIGHT']
                    gi.height_index_up = params['height_index_up']
                    gi.height_index_down = params['height_index_down']
                    gi.BONUS_ITEM_INTERVAL = params['BONUS_ITEM_INTERVAL']
                    gi.game_duration = params['game_duration']

                    player_id = input_box.get_text()  # 获取输入的 ID

                    menu = False
                    game_over = False
                    gi.start_time = time.time()  # 开始游戏时初始化计时

                if quit_button.is_clicked(event):
                    running = False

            elif game_over:
                if back_button.is_clicked(event):
                    menu = True
                    game_over = False

        if menu:
            scaled_menu_background_image = pygame.transform.scale(gi.menu_background_image, (gi.SCREEN_WIDTH, gi.SCREEN_HEIGHT))
            gi.screen.blit(scaled_menu_background_image, [0, 0])
            title_font = pygame.font.SysFont("Comic Sans MS", 60)
            title_text = title_font.render("Fish Game (Squat)", True, color.YELLOW)
            title_rect = title_text.get_rect(center=(gi.SCREEN_WIDTH // 2, gi.SCREEN_HEIGHT // 6))
            gi.screen.blit(title_text, title_rect)
            dropdown.draw(gi.screen)
            adjust_button_position(gi.SCREEN_WIDTH, gi.SCREEN_HEIGHT, dropdown, start_button, quit_button, back_button, input_box)
            if not dropdown.active:
                start_button.draw(gi.screen)
                quit_button.draw(gi.screen)
                input_box.draw(gi.screen)
            pygame.display.flip()

        elif not game_over:
            if gi.initial_phase:
                gi.screen.fill(color.BLACK)
                draw_initial_countdown()
            else:
                try:
                    results, frame = process_camera_frame()
                    
                    if results is not None and frame is not None:
                        print("成功获取摄像头画面")
                        if results.pose_landmarks:
                            print("成功检测到人体姿势")
                        else:
                            print("未检测到人体姿势")
                            
                        capture_all_landmarks(results)
                        shoulder_mid_y = get_shoulder_midpoint_y(results)
                        
                        if shoulder_mid_y is not None:
                            print(f"肩部位置: {shoulder_mid_y}")
                            gi.avatar.update_position(shoulder_mid_y)
                            if gi.calibration_phase:
                                gi.calibrated_height = calibrate_height(shoulder_mid_y)
                                if gi.calibrated_height is not None:
                                    print("校准完成")
                                    gi.calibration_phase = False

                        if not gi.calibration_phase:
                            add_obstacles(gi.SCREEN_HEIGHT, gi.height_index_up, gi.height_index_down)
                            add_bonus_item(gi.SCREEN_HEIGHT)
                            gi.all_sprites.update()

                            if manage_score(feedback_controller):
                                game_over = True
                                final_time = time.time() - gi.start_time

                            if manage_star(star_controller):
                                game_over = True
                                final_time = time.time() - gi.start_time
                        
                        # 传入 frame 和 results 参数
                        draw_and_update_screen(feedback_controller, star_controller, font, font_big, frame, results)
                    else:
                        # 当没有摄像头画面时，仍然更新屏幕，但不传入frame和results
                        draw_and_update_screen(feedback_controller, star_controller, font, font_big)
                        
                except Exception as e:
                    print(f"游戏运行时出错: {e}")
                    draw_and_update_screen(feedback_controller, star_controller, font, font_big)

                clock.tick(30)

                if time.time() >= game_end_time:
                    game_over = True  # 设置为游戏结束
                    final_time = time.time() - gi.start_time  # 保存最终时间

        if game_over and not uploaded:
            draw_game_over_screen(final_time)
            print("Game ended. Saving landmarks to Google Sheets...")
            upload_to_google_sheets(landmarks_list, email_to_share, player_id, "Marine Adventure")
            uploaded = True  # 标记已经上传过数据

    try:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    main()