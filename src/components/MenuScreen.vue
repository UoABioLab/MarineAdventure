<template>
        <div class="menu-screen" v-if="!isGameStarted">
            <!-- 添加背景图片 -->
            <div class="menu-background"></div>
            <!-- 标题 -->
            <h1 class="game-title">{{ t('title') }}</h1>
            
            <!-- 玩家ID输入框 -->
            <!-- <div class="input-box">
                <input 
                    type="text" 
                    v-model="playerId"
                    placeholder="Enter your ID"
                    :class="{ 'active': isInputActive }"
                    @focus="isInputActive = true"
                    @blur="isInputActive = false"
                >
            </div> -->

            <!-- 难度选择 -->
            <div class="dropdown">
                <div class="dropdown-header" @click="toggleDropdown">
                    {{ selectedDifficulty ? t(selectedDifficulty) : t('selectDifficulty') }}
                </div>
                <div class="dropdown-options" v-if="isDropdownOpen">
                    <div 
                        v-for="difficulty in difficulties" 
                        :key="difficulty"
                        class="dropdown-option"
                        @click.stop="selectDifficulty(difficulty)"
                    >
                        {{ t(difficulty) }}
                    </div>
                </div>
            </div>
            <!-- 开始按钮 -->
            <button 
                class="start-button"
                :disabled="!canStartGame"
                @click="startGame"
            >
            {{ t('start') }}
            </button>
            <!-- 退出按钮 -->
            <button 
                class="Mainpage-button"
                @click="backToMain"
            >
            {{ t('back') }}
            </button>
        </div>
    </template>
    <script setup>
    import { t } from '../Language/language.js'
    import { ref, computed } from 'vue'
    const emit = defineEmits(['start-game'])
    const isGameStarted = ref(false)
    const playerId = ref('')
    const selectedDifficulty = ref('')
    const isDropdownOpen = ref(false)
    const isInputActive = ref(false)
    const difficulties = ['easy', 'medium', 'hard']
    // 检查是否可以开始游戏
    const canStartGame = computed(() => {
        // return playerId.value.trim() && selectedDifficulty.value
        return !!selectedDifficulty.value
    })
    // 切换下拉菜单
    const toggleDropdown = () => {
        isDropdownOpen.value = !isDropdownOpen.value
    }
    // 选择难度
    const selectDifficulty = (difficulty) => {
        selectedDifficulty.value = difficulty
        isDropdownOpen.value = false
    }
    // 开始游戏
    const startGame = () => {
        if (canStartGame.value) {
            emit('start-game', {
                // playerId: playerId.value,
                difficulty: selectedDifficulty.value
            })
            isGameStarted.value = true
        }
    }
    // 退出游戏
    const backToMain = () => {
        const referrer = document.referrer;

        // 如果有来源页面，就回去
        if (referrer) {
            window.location.href = referrer;
        } else {
            // 否则根据当前部署自动跳主页
            const isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
            const currentPath = location.pathname.split('/')[1];
            const mainPage = isLocal
                ? '/GameIndex/index.html'
                : `https://${location.hostname}/${currentPath}/`;

            window.location.href = mainPage;
        }
    };
    
    </script>
    <style scoped>
    .menu-screen {
        font-family: 'Arial', sans-serif;
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: none;
        z-index: 10;
    }
    .menu-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-image: url('../assets/images/background_1.jpg');
        background-size: cover;
        background-position: center;
        z-index: -1;
        opacity: 0.8;
    }
    .game-title {
        color: #FFFF00;
        font-size: 70px;
        margin-bottom: 50px;
        text-shadow: 4px 4px 8px rgba(0, 0, 0, 0.5);
    }
    .dropdown {
        width: 300px;
        margin-bottom: 20px;
        position: relative;
    }
    .dropdown-header {
        padding: 30px;
        font-size: 36px;
        font-weight: bold;
        background: #0066cc;
        color: white;
        border-radius: 8px;
        cursor: pointer;
        text-align: center;
    }
    .dropdown-options {
        position: absolute;
        top: 100%;
        left: 0;
        width: 100%;
        background: rgba(0, 0, 0, 0.9);
        border-radius: 8px;
        margin-top: 5px;
        overflow: hidden;
    }
    .dropdown-option {
        padding: 30px;
        font-size: 36px;
        font-weight: bold;
        color: white;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    .dropdown-option:hover {
        background: #0066cc;
    }
    .start-button, .Mainpage-button {
        width: 300px;
        padding: 30px;
        margin: 10px 0;
        border: none;
        border-radius: 8px;
        font-size: 36px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .start-button {
        background: #45f045;
        color: rgb(255, 255, 255);
    }
    .start-button:disabled {
        background: #666666;
        cursor: not-allowed;
    }
    .Mainpage-button {
        background: #cc0000;
        color: white;
        margin-top: 20px;
    }
    .Mainpage-button:hover {
        background: #ff0000;
        transform: scale(1.05);
    }
    </style>
