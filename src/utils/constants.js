// 颜色常量 (从 color.py 转换)
export const COLORS = {
        WHITE: '#FFFFFF',
        BLACK: '#000000',
        RED: '#FF0000',
        GREEN: '#00FF00',
        BLUE: '#0000FF',
        YELLOW: '#FFFF00',
        CYAN: '#00FFFF',
        MAGENTA: '#FF00FF',
        SILVER: '#C0C0C0',
        GRAY: '#808080',
        DARKGRAY: '#646464',
        DARKRED: '#8B0000',
        DARK_GREEN: '#006400',
        OLIVE_GREEN: '#228B22'
    }
    
    // 资源路径
    export const ASSETS = {
        IMAGES: {
            AVATAR_FRAME_0: new URL('../assets/images/avatar_frame_1.png', import.meta.url).href,
            AVATAR_FRAME_1: new URL('../assets/images/avatar_frame_2.png', import.meta.url).href,
            BACKGROUND: new URL('../assets/images/background.png', import.meta.url).href,
            MENU_BACKGROUND: new URL('../assets/images/background_1.jpg', import.meta.url).href,
            BONUS_1: new URL('../assets/images/bonus1.png', import.meta.url).href,
            BONUS_2: new URL('../assets/images/bonus2.png', import.meta.url).href,
            BONUS_3: new URL('../assets/images/bonus3.png', import.meta.url).href,
            OBSTACLE: new URL('../assets/images/obstacle.svg', import.meta.url).href
        },
        SOUNDS: {
            BACKGROUND_MUSIC: new URL('../assets/sounds/backgroundmusic.mp3', import.meta.url).href,
            COLLISION: new URL('../assets/sounds/collision.wav', import.meta.url).href,
            COIN: new URL('../assets/sounds/coin.mp3', import.meta.url).href
        }
    }

    // 游戏配置 (从 initialization.py 转换)
    export const GAME_CONFIG = {
        SCREEN_WIDTH: 1920,
        SCREEN_HEIGHT: 1080,
        OBSTACLE_MIN_GAP: 3,
        OBSTACLE_MAX_GAP: 5,
        BONUS_ITEM_INTERVAL: 5,
        GAME_DURATION: 120,
        INITIAL_LIFE: 3
    }
    
    // 难度设置 (从 DifficultyManager.py 转换)
    export const DIFFICULTY_SETTINGS = {
        EASY: {
            GAME_DURATION: 60,
            SWIM_SPEED: 0.05,
            BASE_CONFIG: {
                top: { baseValue: 0.53, variance: 0.1, minHeight: 20 },
                bottom: { baseValue: 0.32, variance: 0.05, minHeight: 10 }
            }
        },
        MEDIUM: {
            GAME_DURATION: 90,
            SWIM_SPEED: 0.08,
            BASE_CONFIG: {
                top: { baseValue: 0.6, variance: 0.08, minHeight: 20 },
                bottom: { baseValue: 0.35, variance: 0.04, minHeight: 10 }
            }
        },
        HARD: {
            GAME_DURATION: 120,
            SWIM_SPEED: 0.1,
            BASE_CONFIG: {
                top: { baseValue: 0.7, variance: 0.05, minHeight: 20 },
                bottom: { baseValue: 0.4, variance: 0.03, minHeight: 10 }
            }
        }
    }