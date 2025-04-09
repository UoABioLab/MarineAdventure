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
  
  // 游戏配置 (从 initialization.py 转换)
  export const GAME_CONFIG = {
    SCREEN_WIDTH: 800,
    SCREEN_HEIGHT: 600,
    OBSTACLE_MIN_GAP: 3,
    OBSTACLE_MAX_GAP: 5,
    TOP_OBSTACLE_MIN_HEIGHT: 20,
    BOTTOM_OBSTACLE_MIN_HEIGHT: 10,
    BONUS_ITEM_INTERVAL: 5,
    GAME_DURATION: 120,
    INITIAL_LIFE: 3
  }
  
  // 资源路径
  export const ASSETS = {
    IMAGES: {
      AVATAR_FRAME_1: new URL('../assets/images/avatar_frame_1.png', import.meta.url).href,
      AVATAR_FRAME_2: new URL('../assets/images/avatar_frame_2.png', import.meta.url).href,
      BACKGROUND: new URL('../assets/images/background.png', import.meta.url).href,
      MENU_BACKGROUND: new URL('../assets/images/background_1.jpg', import.meta.url).href,
      BONUS_1: new URL('../assets/images/bonus1.png', import.meta.url).href,
      BONUS_2: new URL('../assets/images/bonus2.png', import.meta.url).href,
      BONUS_3: new URL('../assets/images/bonus3.png', import.meta.url).href
    },
    SOUNDS: {
      BACKGROUND_MUSIC: new URL('../assets/sounds/backgroundmusic.mp3', import.meta.url).href,
      COLLISION: new URL('../assets/sounds/collision.wav', import.meta.url).href,
      COIN: new URL('../assets/sounds/coin.mp3', import.meta.url).href
    }
  }
  
  // 难度设置 (从 DifficultyManager.py 转换)
  export const DIFFICULTY_SETTINGS = {
    EASY: {
      OBSTACLE_MIN_GAP: 11,
      OBSTACLE_MAX_GAP: 13,
      TOP_OBSTACLE_MIN_HEIGHT: 30,
      BOTTOM_OBSTACLE_MIN_HEIGHT: 30,
      HEIGHT_INDEX_UP: 5,
      HEIGHT_INDEX_DOWN: 2,
      BONUS_ITEM_INTERVAL: 30,
      TOP_COUNT_PER_TIME: { LOW: 1, HIGH: 2 },
      BOTTOM_COUNT_PER_TIME: { LOW: 1, HIGH: 2 },
      GAME_DURATION: 90
    },
    MEDIUM: {
      OBSTACLE_MIN_GAP: 8,
      OBSTACLE_MAX_GAP: 10,
      TOP_OBSTACLE_MIN_HEIGHT: 30,
      BOTTOM_OBSTACLE_MIN_HEIGHT: 30,
      HEIGHT_INDEX_UP: 3,
      HEIGHT_INDEX_DOWN: 3.5,
      BONUS_ITEM_INTERVAL: 20,
      TOP_COUNT_PER_TIME: { LOW: 1, HIGH: 3 },
      BOTTOM_COUNT_PER_TIME: { LOW: 1, HIGH: 2 },
      GAME_DURATION: 120
    },
    HARD: {
      OBSTACLE_MIN_GAP: 4,
      OBSTACLE_MAX_GAP: 6,
      TOP_OBSTACLE_MIN_HEIGHT: 30,
      BOTTOM_OBSTACLE_MIN_HEIGHT: 30,
      HEIGHT_INDEX_UP: 0.5,
      HEIGHT_INDEX_DOWN: 5,
      BONUS_ITEM_INTERVAL: 10,
      TOP_COUNT_PER_TIME: { LOW: 1, HIGH: 3 },
      BOTTOM_COUNT_PER_TIME: { LOW: 1, HIGH: 2 },
      GAME_DURATION: 180
    }
  }