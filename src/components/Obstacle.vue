<template>
  <canvas ref="obstacleCanvas" class="obstacles"></canvas>
</template>
<script setup>
import { ref, onMounted, watch } from 'vue'
import { COLORS } from '../utils/constants'
const props = defineProps({
  obstacles: {
    type: Array,
    required: true,
    default: () => []
  },
  assets: {
    type: Object,
    required: false
  }
})
const obstacleCanvas = ref(null)
// 绘制障碍物
const drawObstacles = (ctx) => {
  if (!obstacleCanvas.value) return  // 添加空值检查
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
  props.obstacles.forEach(obstacle => {
    // 使用红色方块作为障碍物
    ctx.fillStyle = COLORS.RED
    ctx.fillRect(
      obstacle.x,
      obstacle.y,
      obstacle.width,
      obstacle.height
    )
  })
}
// 监听障碍物数组变化
watch(() => props.obstacles, () => {
  if (obstacleCanvas.value) {
    const ctx = obstacleCanvas.value.getContext('2d')
    drawObstacles(ctx)
  }
}, { deep: true })
onMounted(() => {
  try {
    if (obstacleCanvas.value) {
      obstacleCanvas.value.width = window.innerWidth
      obstacleCanvas.value.height = window.innerHeight
      // 初始绘制
      const ctx = obstacleCanvas.value.getContext('2d')
      drawObstacles(ctx)
      // 监听窗口大小变化
      window.addEventListener('resize', () => {
        if (obstacleCanvas.value) {  // 添加空值检查
          obstacleCanvas.value.width = window.innerWidth
          obstacleCanvas.value.height = window.innerHeight
          drawObstacles(obstacleCanvas.value.getContext('2d'))
        }
      })
    }
  } catch (error) {
    console.error('Error in Obstacle component:', error)
  }
})
</script>
<style scoped>
.obstacles {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 1;
}
</style>
