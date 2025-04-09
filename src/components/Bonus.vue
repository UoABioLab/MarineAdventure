<template>
    <canvas ref="bonusCanvas" class="bonus"></canvas>
  </template>
  <script setup>
  import { ref, onMounted, watch } from 'vue'
  const props = defineProps({
    bonusItems: {
      type: Array,
      required: true,
      default: () => []
    },
    assets: {
      type: Object,
      required: false
    }
  })
  const emit = defineEmits(['collect'])
  const bonusCanvas = ref(null)
  // 绘制奖励物品
  const drawBonusItems = (ctx) => {
    if (!bonusCanvas.value) return
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    props.bonusItems.forEach(item => {
      if (props.assets && props.assets.images) {
        // 根据类型选择不同的贝壳图片
        const imageKey = `BONUS_${item.type}`
        const image = props.assets.images[imageKey]
        if (image) {
          ctx.drawImage(
            image,
            item.x,
            item.y,
            80,  // 贝壳宽度
            80   // 贝壳高度
          )
        }
      }
    })
  }
  // 监听奖励物品数组变化
  watch(() => props.bonusItems, () => {
    if (bonusCanvas.value) {
      const ctx = bonusCanvas.value.getContext('2d')
      drawBonusItems(ctx)
    }
  }, { deep: true })
  onMounted(() => {
    if (bonusCanvas.value) {
      bonusCanvas.value.width = window.innerWidth
      bonusCanvas.value.height = window.innerHeight
      const ctx = bonusCanvas.value.getContext('2d')
      drawBonusItems(ctx)
      window.addEventListener('resize', () => {
        if (bonusCanvas.value) {
          bonusCanvas.value.width = window.innerWidth
          bonusCanvas.value.height = window.innerHeight
          drawBonusItems(bonusCanvas.value.getContext('2d'))
        }
      })
    }
  })
  </script>
  <style scoped>
  .bonus {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
  }
  </style>
