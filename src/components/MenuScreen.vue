<template>
    <div class="menu-screen" v-if="!isGameStarted">
      <!-- 添加背景图片 -->
      <div class="menu-background"></div>
      <!-- 标题 -->
      <h1 class="game-title">Fish Game (Squat)</h1>
      <!-- 玩家ID输入框 -->
      <div class="input-box">
        <input 
          type="text" 
          v-model="playerId"
          placeholder="Enter your ID"
          :class="{ 'active': isInputActive }"
          @focus="isInputActive = true"
          @blur="isInputActive = false"
        >
      </div>
      <!-- 难度选择 -->
      <div class="dropdown">
        <div class="dropdown-header" @click="toggleDropdown">
          {{ selectedDifficulty || 'Select Difficulty' }}
        </div>
        <div class="dropdown-options" v-if="isDropdownOpen">
          <div 
            v-for="difficulty in difficulties" 
            :key="difficulty"
            class="dropdown-option"
            @click.stop="selectDifficulty(difficulty)"
          >
            {{ difficulty }}
          </div>
        </div>
      </div>
      <!-- 开始按钮 -->
      <button 
        class="start-button"
        :disabled="!canStartGame"
        @click="startGame"
      >
        Start Game
      </button>
      <!-- 退出按钮 -->
      <button 
        class="quit-button"
        @click="quitGame"
      >
        Quit
      </button>
    </div>
  </template>
  <script setup>
  import { ref, computed } from 'vue'
  const emit = defineEmits(['start-game', 'quit-game'])
  const isGameStarted = ref(false)
  const playerId = ref('')
  const selectedDifficulty = ref('')
  const isDropdownOpen = ref(false)
  const isInputActive = ref(false)
  const difficulties = ['Easy', 'Medium', 'Hard']
  // 检查是否可以开始游戏
  const canStartGame = computed(() => {
    return playerId.value.trim() && selectedDifficulty.value
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
        playerId: playerId.value,
        difficulty: selectedDifficulty.value
      })
      isGameStarted.value = true
    }
  }
  // 退出游戏
  const quitGame = () => {
    window.location.href = 'https://uoabiolab.github.io/MarineAdventure/'
  }
  </script>
  <style scoped>
  .menu-screen {
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
    font-size: 48px;
    margin-bottom: 50px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
  }
  .input-box {
    width: 300px;
    margin-bottom: 20px;
  }
  .input-box input {
    width: 100%;
    padding: 12px;
    border: 2px solid #0066cc;
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.5);
    color: white;
    font-size: 16px;
    transition: all 0.3s ease;
  }
  .input-box input.active {
    border-color: #00ff00;
    box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
  }
  .dropdown {
    width: 300px;
    margin-bottom: 20px;
    position: relative;
  }
  .dropdown-header {
    padding: 12px;
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
    padding: 12px;
    color: white;
    cursor: pointer;
    transition: background 0.3s ease;
  }
  .dropdown-option:hover {
    background: #0066cc;
  }
  .start-button, .quit-button {
    width: 300px;
    padding: 12px;
    margin: 10px 0;
    border: none;
    border-radius: 8px;
    font-size: 18px;
    cursor: pointer;
    transition: all 0.3s ease;
  }
  .start-button {
    background: #00cc00;
    color: white;
  }
  .start-button:disabled {
    background: #666666;
    cursor: not-allowed;
  }
  .quit-button {
    background: #cc0000;
    color: white;
    margin-top: 20px;
  }
  .quit-button:hover {
    background: #ff0000;
    transform: scale(1.05);
  }
  </style>
