<template>
    <div class="game-canvas">
        <canvas ref="canvas"></canvas>
        <!-- 倒计时显示 -->
        <div class="countdown" v-if="showCountdown">
            <div class="countdown-text">
                {{ countdownText }}
            </div>
        </div>
        <!-- 游戏内容 -->
        <template v-if="isPlaying">
            <GameObject 
                :gameObjects="gameObjects"
                :assets="gameAssets"
            />
            <div class="score-display">
                <div class="life">{{ t('lives') }}: {{ life }}</div>
                <div class="score"> {{ t('score') }}: {{ score }}</div>
                <div class="time" :class="{ 'time-warning': gameTime <= 30 }">
                    {{ t('countTime') }}: {{ gameTime }}s
                </div>
            </div>
        </template>
        <!-- 游戏结束画面 -->
        <div class="game-over" v-if="!isPlaying && gameStarted && !showCountdown">
            <div class="game-over-content">
                <h1>Game Over</h1>
                <div class="stats">
                    <p>{{ t('timePlayed') }}: {{ GAME_TIME_LIMIT - gameTime }}</p>
                    <p>{{ t('finalScore') }}: {{ score }}</p>
                    <p>{{ t('finalLife') }}: {{ life }}</p>
                </div>
                <div class="buttons">
                    <button @click="startGame">{{ t('retry')}}</button>
                    <button @click="backToMenu">{{ t('back')}}</button>
                </div>
            </div>
        </div>
        <!-- 添加校准阶段提示 -->
        <div class="calibration" v-if="isCalibrating">
            <div class="calibration-text">
                {{ calibrationText }}
            </div>
            <div class="calibration-progress">
                <div 
                    class="progress-bar" 
                    :style="{ width: `${(calibrationProgress / CALIBRATION_SAMPLES) * 100}%` }"
                ></div>
            </div>
        </div>
        <!-- 添加分数动画 -->
        <div class="score-animations">
            <transition-group name="score-popup">
                <div 
                    v-for="animation in scoreAnimations" 
                    :key="animation.id"
                    class="score-popup"
                    :style="{ 
                        left: `${animation.x}px`, 
                        top: `${animation.y}px`,
                        color: animation.value > 0 ? '#4CAF50' : '#FF0000'
                    }"
                >
                    {{ animation.value > 0 ? '+' : '' }}{{ animation.value }}
                </div>
            </transition-group>
        </div>
    </div>
</template>
<script setup>
import { t } from '../Language/language.js'
import { ref, onMounted, onUnmounted, watch } from 'vue'
// import Avatar from './Avatar.vue'
import GameObject from './GameObject.vue'
// import Bonus from './Bonus.vue'
import { GAME_CONFIG, DIFFICULTY_SETTINGS } from '../utils/constants'
const emit = defineEmits(['game-over', 'back-to-menu'])
// 游戏状态
const canvas = ref(null)
const gameAssets = ref(null)
const gameObjects = ref({})
// const avatarPosition = ref({ x: 0, y: 0 })
const isPlaying = ref(false)
const life = ref(GAME_CONFIG.INITIAL_LIFE)
const score = ref(0)
const calibratedHeight = ref(null)
let gameLoop = null
let lastObstacleTime = 0
let obstacleToggle = 'top'
let lastBonusTime = 0
// 添加碰撞冷却时间
let collisionCooldown = false
const COLLISION_COOLDOWN_TIME = 1000 // 1秒冷却时间
// 添加游戏时间和状态
const gameStarted = ref(false)
const gameStartTime = ref(0)
const gameTime = ref()
let lastUpdateTime = null
let gameTimer = null
// 添加倒计时状态
const showCountdown = ref(false)
const countdownText = ref('')
// 添加游戏时间限制常量
let GAME_TIME_LIMIT = 120 // 默认值，之后在 initGame 里动态赋值
// 添加难度属性
const difficulty = ref('Medium') // 默认难度
let DIFFICULTY_CONFIG = DIFFICULTY_SETTINGS.MEDIUM
// 添加校准相关常量
const CALIBRATION_DURATION = 3000 // 3秒校准时间
const CALIBRATION_SAMPLES = 30 // 采样次数
const CALIBRATION_INTERVAL = CALIBRATION_DURATION / CALIBRATION_SAMPLES
// 添加校准相关状态
const isCalibrating = ref(false)
const calibrationText = ref('')
const calibrationProgress = ref(0)
const calibrationSamples = ref([])
// 添加最小横向间距常量
const MIN_OBSTACLE_X_GAP = 200    // 最小横向间距，可以根据需要调整
// 添加分数动画数组
const scoreAnimations = ref([])
let animationId = 0
// 添加背景音乐状态
let backgroundMusic = null
// 初始化游戏
const initGame = () => {
    // 绘制背景
    if (canvas.value && gameAssets.value) {
        const ctx = canvas.value.getContext('2d')
        ctx.drawImage(
            gameAssets.value.images.BACKGROUND,
            0,
            0,
            canvas.value.width,
            canvas.value.height
        )
    }
    life.value = GAME_CONFIG.INITIAL_LIFE
    score.value = 0
    lastObstacleTime = Date.now()
    gameStarted.value = true
    isPlaying.value = true
    gameStartTime.value = Date.now()
    
    GAME_TIME_LIMIT = DIFFICULTY_CONFIG.GAME_DURATION || 120 // 默认2分钟，如果难度配置中没有设置则使用默认值
    console.log('Game time limit:', GAME_TIME_LIMIT)
    // 根据难度设置游戏时间
    gameTime.value = DIFFICULTY_CONFIG.GAME_DURATION
    
    // 修改计时器，现在是倒计时
    gameTimer = setInterval(() => {
        if (gameTime.value > 0) {
            gameTime.value--
            // 当时间到达0时结束游戏
            if (gameTime.value <= 0) {
                gameOver()
            }
        }
    }, 1000)
    
    // 播放背景音乐
    if (gameAssets.value?.sounds?.BACKGROUND_MUSIC) {
        backgroundMusic = gameAssets.value.sounds.BACKGROUND_MUSIC
        backgroundMusic.loop = true
        backgroundMusic.volume = 0.5
        backgroundMusic.play()
    }
    
    startGameLoop()
}
// 游戏主循环
const startGameLoop = () => {
    lastUpdateTime = Date.now()
    const loop = () => {
        try {
            if (!isPlaying.value) return
            // 清除画布
            // const ctx = canvas.value.getContext('2d')
            // ctx.clearRect(0, 0, canvas.value.width, canvas.value.height)
            // // 重新绘制背景
            // if (gameAssets.value && gameAssets.value.images) {
            //     ctx.drawImage(
            //         gameAssets.value.images.BACKGROUND,
            //         0,
            //         0,
            //         canvas.value.width,
            //         canvas.value.height
            //     )
            // }
            updateGame()
            gameLoop = requestAnimationFrame(loop)
            lastUpdateTime = Date.now()
        } catch (error) {
            console.error('Error in game loop:', error)
            // 如果发生错误，尝试重新启动游戏循环
            gameLoop = requestAnimationFrame(loop)
            lastUpdateTime = Date.now()
        }
    }
    gameLoop = requestAnimationFrame(loop)
}
// 更新游戏状态
const updateGame = () => {
    if (!calibratedHeight.value) return
    
    const currentTime = Date.now()
    // 更新障碍物生成
    if (currentTime - lastObstacleTime > GAME_CONFIG.OBSTACLE_MIN_GAP * 1000) {
        const lastObstacle = GAME_CONFIG.SCREEN_WIDTH - (currentTime - lastObstacleTime) * DIFFICULTY_CONFIG.SWIM_SPEED + 50
        const minGap = Math.floor(Math.random() * (500 - 350 + 1)) + 350
        const canAddObstacle = lastObstacle + minGap <= GAME_CONFIG.SCREEN_WIDTH
        
        if (canAddObstacle) {
            addObstacle()
            lastObstacleTime = currentTime
        }
    }
    
    // 更新贝壳生成（从3秒改为8秒）
    if (currentTime - lastBonusTime > 8000) {    // 改为8000毫秒
        addBonusItem()
        lastBonusTime = currentTime
    }
    
    updateGameObjects()
}
// 修改添加障碍物的函数，调整生成位置
const addObstacle = () => {
    // 根据难度设置高度系数
    const baseValue = GAME_CONFIG.SCREEN_HEIGHT * DIFFICULTY_CONFIG.BASE_CONFIG[obstacleToggle].baseValue
    const variance = GAME_CONFIG.SCREEN_HEIGHT * DIFFICULTY_CONFIG.BASE_CONFIG[obstacleToggle].variance
    
    const minHeight = Math.max(baseValue - variance, DIFFICULTY_CONFIG.BASE_CONFIG[obstacleToggle].minHeight)
    const maxHeight = Math.min(baseValue + variance, GAME_CONFIG.SCREEN_HEIGHT - 50)

    const height = Math.floor(Math.random() * (maxHeight - minHeight) + minHeight)
    const newObstacle = {
        x: GAME_CONFIG.SCREEN_WIDTH,
        y: obstacleToggle === 'top' ? height-1000 : GAME_CONFIG.SCREEN_HEIGHT - height,
        width: 50,
        height: 1000,
        type: obstacleToggle,
        collidable: true,
        image: `OBSTACLE`,
        frames: 1
    }
    gameObjects.value.obstacles.push(newObstacle)
    obstacleToggle = obstacleToggle === 'top' ? 'bottom' : 'top'
}
// 修改生成奖励物品的函数
const addBonusItem = () => {
    const width = 80
    const height = 80
    
    // 只在校准完成后生成贝壳
    if (!calibratedHeight.value) return

    if (lastObstacleTime == 0) return
    
    // 找到最右边的障碍物
    const lastObstacle = GAME_CONFIG.SCREEN_WIDTH - (Date.now() - lastObstacleTime) * DIFFICULTY_CONFIG.SWIM_SPEED + 50
    
    // 在最右边的障碍物和下一个障碍物之间的空隙中生成贝壳
    const minGap = Math.floor(Math.random() * (500 - 350 + 1)) + 350 // 350-500之间的随机值
    const bonusX = lastObstacle + (minGap / 2)
    
    // 生成Y坐标，避开障碍物的高度范围
    const minY = 100
    const maxY = GAME_CONFIG.SCREEN_HEIGHT - height - 100
    const testY = Math.random() * (maxY - minY) + minY
    
    // 创建贝壳
    const type = Math.floor(Math.random() * 3) + 1
    gameObjects.value.bonusItems.push({
        x: bonusX,
        y: testY,
        width: width,
        height: height,
        collidable: true,
        image: `BONUS_${type}`,
        frames: 1
    })
}
// 更新障碍物位置并处理得分
const updateGameObjects = () => {
    const deltaTime = Date.now() - lastUpdateTime
    gameObjects.value.obstacles = gameObjects.value.obstacles.filter(obstacle => {
        obstacle.x -= deltaTime * DIFFICULTY_CONFIG.SWIM_SPEED
        if (obstacle.collidable){
            if (obstacle.x + obstacle.width < gameObjects.value.player.x) {
                obstacle.collidable = false
                addScore(obstacle.x, obstacle.y)
            }else if (checkCollision(gameObjects.value.player, obstacle)) {
                obstacle.collidable = false
                handleCollision(obstacle.x, obstacle.y)
            }
        }
        return obstacle.x + obstacle.width > 0
    })
    gameObjects.value.bonusItems = gameObjects.value.bonusItems.filter(item => {
        item.x -= deltaTime * DIFFICULTY_CONFIG.SWIM_SPEED
        if (item.collidable) {
            if (checkCollision(gameObjects.value.player, item)) {
                addScore(item.x,item.y)
                return false
            }
        }
        return item.x + item.width > 0
    })
}
// 碰撞检测辅助函数
const checkCollision = (rect1, rect2) => {
    return rect1.x < rect2.x + rect2.width &&
                 rect1.x + rect1.width > rect2.x &&
                 rect1.y < rect2.y + rect2.height &&
                 rect1.y + rect1.height > rect2.y
}
// 处理碰撞
const handleCollision = (x, y) => {
    if (collisionCooldown) return
    if (life.value > 0) {
        life.value--
        // 使用正确的位置显示扣分动画
        showScoreAnimation(x, y, -1)
        // 播放碰撞音效
        if (gameAssets.value?.sounds?.COLLISION) {
            gameAssets.value.sounds.COLLISION.play()
        }
        // 设置碰撞冷却
        collisionCooldown = true
        setTimeout(() => {
            collisionCooldown = false
        }, COLLISION_COOLDOWN_TIME)
    }
    if (life.value <= 0) {
        gameOver()
    }
}
// 添加分数增加函数
const addScore = (x, y) => {
    score.value += 10
    // 显示得分动画
    showScoreAnimation(x, y, +10)
    // 播放得分音效
    if (gameAssets.value?.sounds?.COIN) {
        gameAssets.value.sounds.COIN.play()
    }
}
// 游戏结束
const gameOver = () => {
    isPlaying.value = false
    // 停止游戏循环和计时器
    if (gameLoop) {
        cancelAnimationFrame(gameLoop)
    }
    if (gameTimer) {
        clearInterval(gameTimer)
    }
    // 发送游戏结束事件，包含最终得分和剩余时间
    emit('game-over', {
        score: score.value,
        time: GAME_TIME_LIMIT - gameTime.value,
        life: life.value
    })
}
// 返回菜单
const backToMenu = () => {
    // 重置所有游戏状态
    resetGame()
    // 重置难度
    difficulty.value = 'Medium'
    // 发送返回菜单事件
    emit('back-to-menu')
}
// 组件卸载时清理
onUnmounted(() => {
    if (gameTimer) {
        clearInterval(gameTimer)
    }
    // 停止并清理背景音乐
    if (backgroundMusic) {
        backgroundMusic.pause()
        backgroundMusic = null
    }
})
// 更新角色位置（由姿势检测触发）
const updateAvatarPosition = (shoulderY) => {
    if (isCalibrating.value) {
        // 校准阶段直接使用原始值
        gameObjects.value.player.x = GAME_CONFIG.SCREEN_WIDTH * 0.4
        gameObjects.value.player.y = shoulderY
    } else if (calibratedHeight.value !== null) {
        // 游戏阶段使用归一化后的值
        const normalizedY = shoulderY / GAME_CONFIG.SCREEN_HEIGHT    // 转换为0-1之间的比例
        
        // 计算相对于校准位置的偏移量
        const relativeY = normalizedY - calibratedHeight.value
        
        // 添加放大系数，增加移动幅度
        const amplificationFactor = 2.5    // 可以调整这个系数来改变灵敏度
        const amplifiedY = relativeY * GAME_CONFIG.SCREEN_HEIGHT * amplificationFactor
        
        // 计算新位置，以屏幕中心为基准
        const newY = GAME_CONFIG.SCREEN_HEIGHT / 2 + amplifiedY
        
        // 限制在屏幕范围内
        const minY = 0
        const maxY = GAME_CONFIG.SCREEN_HEIGHT - 80
        
        // 平滑移动
        const currentY = gameObjects.value.player.y
        const smoothFactor = 0.2    // 平滑系数，可以调整
        const targetY = Math.max(minY, Math.min(maxY, newY))
        
        // 使用线性插值实现平滑移动
        gameObjects.value.player.x = GAME_CONFIG.SCREEN_WIDTH * 0.4
        gameObjects.value.player.y = currentY + (targetY - currentY) * smoothFactor
        // console.log(gameObjects.value.player.y)
    }
}
// 设置游戏资源
const setGameAssets = (assets, gameDifficulty) => {
    gameAssets.value = assets
    difficulty.value = gameDifficulty
    DIFFICULTY_CONFIG = DIFFICULTY_SETTINGS[gameDifficulty.toUpperCase()]
}
// 重置游戏状态
const resetGame = () => {
    isPlaying.value = false
    gameStarted.value = false
    showCountdown.value = false
    life.value = GAME_CONFIG.INITIAL_LIFE
    score.value = 0
    gameObjects.value = {player:{x: 0, y: 0, width:80, height:80, image:`AVATAR`, frames:2},obstacles:[],bonusItems:[]}
    gameTime.value = GAME_TIME_LIMIT
    calibratedHeight.value = null    // 重置校准高度
    if (gameLoop) {
        cancelAnimationFrame(gameLoop)
    }
    if (gameTimer) {
        clearInterval(gameTimer)
    }
    // 停止背景音乐
    if (backgroundMusic) {
        backgroundMusic.pause()
        backgroundMusic.currentTime = 0
    }
}
// 修改开始游戏函数
const startGame = async () => {
    // 重置游戏状态
    resetGame()
    // 显示倒计时
    showCountdown.value = true
    countdownText.value = 'Ready!'
    // Ready阶段
    await new Promise(resolve => setTimeout(resolve, 2000))
    // 倒计时阶段
    for(let i = 3; i > 0; i--) {
        countdownText.value = i.toString()
        await new Promise(resolve => setTimeout(resolve, 1000))
    }
    // 开始校准阶段
    await startCalibration()
    // 开始游戏
    showCountdown.value = false
    initGame()
}
// 添加校准阶段函数
const startCalibration = async () => {
    isCalibrating.value = true
    calibrationText.value = t('calibrate')
    calibrationProgress.value = 0
    calibrationSamples.value = []
    // 等待校准完成
    await new Promise(resolve => {
        let sampleCount = 0
        const calibrationInterval = setInterval(() => {
            if (sampleCount >= CALIBRATION_SAMPLES) {
                clearInterval(calibrationInterval)
                finishCalibration()
                resolve()
            } else if (gameObjects.value.player.y !== undefined) {
                calibrationSamples.value.push(gameObjects.value.player.y)
                calibrationProgress.value = ++sampleCount
            }
        }, CALIBRATION_INTERVAL)
    })
    isCalibrating.value = false
}
// 添加完成校准函数
const finishCalibration = () => {
    // 计算平均值
    const sum = calibrationSamples.value.reduce((a, b) => a + b, 0)
    const average = sum / calibrationSamples.value.length
    // 归一化处理
    calibratedHeight.value = average / GAME_CONFIG.SCREEN_HEIGHT
    console.log('Calibrated height:', calibratedHeight.value)
}
// 添加 onMounted 钩子来设画布尺寸
onMounted(() => {
    if (canvas.value) {
        if (window.innerWidth/window.innerHeight > GAME_CONFIG.SCREEN_WIDTH/GAME_CONFIG.SCREEN_HEIGHT){
            canvas.value.height = window.innerHeight
            canvas.value.width = GAME_CONFIG.SCREEN_WIDTH * window.innerHeight / GAME_CONFIG.SCREEN_HEIGHT
        }else{
            canvas.value.width = window.innerWidth
            canvas.value.height = GAME_CONFIG.SCREEN_HEIGHT * window.innerWidth / GAME_CONFIG.SCREEN_WIDTH
        }
        // 监听窗口大小变化
        window.addEventListener('resize', () => {
            if (window.innerWidth/window.innerHeight > GAME_CONFIG.SCREEN_WIDTH/GAME_CONFIG.SCREEN_HEIGHT){
                canvas.value.height = window.innerHeight
                canvas.value.width = GAME_CONFIG.SCREEN_WIDTH * window.innerHeight / GAME_CONFIG.SCREEN_HEIGHT
            }else{
                canvas.value.width = window.innerWidth
                canvas.value.height = GAME_CONFIG.SCREEN_HEIGHT * window.innerWidth / GAME_CONFIG.SCREEN_WIDTH
            }
            // 重新绘制背景
            if (gameAssets.value) {
                const ctx = canvas.value.getContext('2d')
                ctx.drawImage(
                    gameAssets.value.images.BACKGROUND,
                    0,
                    0,
                    canvas.value.width,
                    canvas.value.height
                )
            }
        })
    }
})
// 添加显示分数动画函数
const showScoreAnimation = (x, y, value) => {
    const animation = {
        id: animationId++,
        x,
        y,
        value
    }
    scoreAnimations.value.push(animation)
    setTimeout(() => {
        scoreAnimations.value = scoreAnimations.value.filter(a => a.id !== animation.id)
    }, 1000)
}
defineExpose({
    canvas,
    setGameAssets,
    startGame,
    updateAvatarPosition,
    backToMenu,
    resetGame
})
</script>
<style scoped>
.game-canvas {
    margin: auto;
    z-index: 0;
    overflow: hidden;
}
canvas {
    margin: auto;
    overflow: hidden;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}
.score-display {
    position: fixed;
    top: 20px;
    left: 20px;
    font-size: 48px;
    color: white;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    z-index: 1000;
    background: rgba(0, 0, 0, 0.5);
    padding: 20px;
    border-radius: 10px;
    backdrop-filter: blur(5px);
}
.life {
    margin-bottom: 20px;
    color: #FF6B6B;
    font-weight: bold;
    text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
}
.score {
    color: #4CAF50;
    margin-bottom: 20px;
    font-weight: bold;
    text-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
}
.collision-indicator {
    position: absolute;
    color: red;
    font-size: 32px;
    font-weight: bold;
    animation: fadeUp 1s ease-out;
    opacity: 0;
}
@keyframes fadeUp {
    0% {
        transform: translateY(0);
        opacity: 1;
    }
    100% {
        transform: translateY(-30px);
        opacity: 0;
    }
}
.game-over {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 100;
}
.game-over-content {
    background: rgba(143, 212, 245, 0.9);
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    border: 2px solid rgba(255, 255, 255, 0.1);
}
.game-over h1 {
    color: #FF0000;
    font-size: 64px;
    font-weight: bold;
    margin-bottom: 30px;
    text-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
}
.stats {
    margin-bottom: 30px;
}
.stats p {
    color: white;
    font-size: 48px;
    font-weight: bold;
    margin: 10px 0;
}
.buttons {
    display: flex;
    gap: 20px;
    justify-content: center;
}
.buttons button {
    padding: 15px 30px;
    font-size: 36px;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
}
.buttons button:first-child {
    background: #4CAF50;
    color: white;
}
.buttons button:last-child {
    background: #2196F3;
    color: white;
}
.buttons button:hover {
    transform: translateY(-2px);
    filter: brightness(1.1);
}
.time {
    color: #FFD700;
    font-weight: bold;
    text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
}
.countdown {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background: rgba(0, 0, 0, 0.7);
    z-index: 50;
}
.countdown-text {
    font-size: 72px;
    color: white;
    text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
}
.time-warning {
    color: #FF0000 !important;
    animation: blink 1s infinite;
}
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.calibration {
    z-index: 1100; /* 高于其他所有层 */
    position: fixed;
    top: 30%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(161, 233, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.calibration-text {
    color: rgb(0, 0, 0);
    font-size: 48px;
    margin-bottom: 20px;
    text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.7); /* ✅ 添加阴影增强可读性 */
    font-weight: bold; /* 可选，增加清晰度 */
}
.calibration-progress {
    width: 100%;;
    height: 20px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    overflow: hidden;
}
.progress-bar {
    height: 100%;
    background: #4CAF50;
    transition: width 0.1s linear;
}
.score-popup {
    position: fixed;
    left: 50% !important;    /* 强制居中 */
    top: 50% !important;     /* 强制居中 */
    transform: translate(-50%, -50%);
    font-size: 48px;    /* 增大字体 */
    font-weight: bold;
    pointer-events: none;
    animation: float-up 1s ease-out;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    z-index: 1000;
}
.score-popup.positive {
    color: #4CAF50;
}
.score-popup.negative {
    color: #FF0000;
}
@keyframes float-up {
    0% {
        transform: translate(-50%, -50%) scale(1);
        opacity: 1;
    }
    100% {
        transform: translate(-50%, -100%) scale(1.5);
        opacity: 0;
    }
}
.score-popup-enter-active,
.score-popup-leave-active {
    transition: all 1s ease-out;
}
.score-popup-enter-from,
.score-popup-leave-to {
    opacity: 0;
    transform: translateY(20px);
}
</style>
