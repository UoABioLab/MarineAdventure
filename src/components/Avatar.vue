<template>
    <canvas ref="avatarCanvas" class="avatar"></canvas>
  </template>
  <script setup>
  import { ref, onMounted, watch } from 'vue'
  import { GAME_CONFIG } from '../utils/constants'
  const props = defineProps({
    position: {
      type: Object,
      required: true,
      default: () => ({ x: 0, y: 0 })
    },
    assets: {
      type: Object,
      required: true
    }
  })
  const avatarCanvas = ref(null)
  let currentFrame = 0
  let lastUpdateTime = 0
  const ANIMATION_SPEED = 100 // 动画更新间隔(ms)
  // 更新角色动画
  const updateAnimation = (timestamp) => {
    if (timestamp - lastUpdateTime > ANIMATION_SPEED) {
      currentFrame = (currentFrame + 1) % 2
      lastUpdateTime = timestamp
    }
  }
  // 绘制角色
  const drawAvatar = (ctx) => {
    const currentImage = currentFrame === 0 ? 
      props.assets.images.AVATAR_FRAME_1 : 
      props.assets.images.AVATAR_FRAME_2
    ctx.drawImage(
      currentImage,
      props.position.x,
      props.position.y,
      80, // 宽度
      80  // 高度
    )
  }
  // 动画循环
  const animate = (timestamp) => {
    if (!avatarCanvas.value) return
    const ctx = avatarCanvas.value.getContext('2d')
    ctx.clearRect(0, 0, avatarCanvas.value.width, avatarCanvas.value.height)
    updateAnimation(timestamp)
    drawAvatar(ctx)
    requestAnimationFrame(animate)
  }
  // 监听位置变化
  watch(() => props.position, (newPos) => {
    if (!avatarCanvas.value) return
    console.log('Avatar position updated:', newPos) // 添加日志
    const ctx = avatarCanvas.value.getContext('2d')
    ctx.clearRect(0, 0, avatarCanvas.value.width, avatarCanvas.value.height)
    drawAvatar(ctx)
  }, { deep: true, immediate: true })
  onMounted(() => {
    if (avatarCanvas.value) {
      avatarCanvas.value.width = window.innerWidth
      avatarCanvas.value.height = window.innerHeight
      requestAnimationFrame(animate)
    }
  })
  </script>
  <style scoped>
  .avatar {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
  </style>
