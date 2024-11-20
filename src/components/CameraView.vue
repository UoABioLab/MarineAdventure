<template>
  <div class="camera-container" v-show="showCamera">
    <div class="camera-wrapper">
      <!-- 摄像头视频元素 -->
      <video
        ref="videoElement"
        class="camera-video"
        autoplay
        playsinline
      ></video>
      <!-- 姿势检测画布 -->
      <canvas
        ref="canvasElement"
        class="pose-canvas"
      ></canvas>
      <!-- 状态显示 -->
      <div class="status-overlay">
        <h3 class="camera-title">Camera View</h3>
        <div 
          :class="['status-text', { 'detected': isPoseDetected }]"
        >
          {{ isPoseDetected ? 'Pose Detected' : 'No Pose Detected' }}
        </div>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
const Pose = window.Pose || null
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'

const emit = defineEmits(['shoulder-position'])
const videoElement = ref(null)
const canvasElement = ref(null)
const isPoseDetected = ref(false)
let pose = null
let videoStream = null
const props = defineProps({
  showCamera: {
    type: Boolean,
    default: false
  }
})
// 初始化 TensorFlow
const initTF = async () => {
  try {
    await tf.ready()
    await tf.setBackend('webgl')
    console.log('TensorFlow initialized successfully')
  } catch (error) {
    console.error('Error initializing TensorFlow:', error)
  }
}
// 初始化摄像头
const initCamera = async () => {
  try {
    // 先初始化 TensorFlow
    await initTF()
    
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 640,
        height: 480,
        frameRate: 30
      }
    })
    if (videoElement.value) {
      videoElement.value.srcObject = stream
      videoStream = stream
      await initPoseDetection()
    }
  } catch (error) {
    console.error('Error accessing camera:', error)
  }
}
// 初始化姿势检测
const initPoseDetection = async () => {
  if (!Pose) {
    console.error('MediaPipe Pose not loaded')
    return
  }
  
  try {
    // 确保 TensorFlow 已初始化
    await initTF()
    
    pose = new Pose({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
      }
    })
    
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
    
    pose.onResults(onResults)
    startDetection()
  } catch (error) {
    console.error('Error initializing Pose:', error)
  }
}
// 开始检测循环
const startDetection = async () => {
  if (!videoElement.value || !pose) return
  try {
    await pose.send({ image: videoElement.value })
  } catch (error) {
    console.error('Error in pose detection:', error)
  }
  requestAnimationFrame(startDetection)
}
// 处理检测结果
const onResults = (results) => {
  if (!canvasElement.value) return
  const canvas = canvasElement.value
  const ctx = canvas.getContext('2d')
  canvas.width = videoElement.value.videoWidth
  canvas.height = videoElement.value.videoHeight
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  isPoseDetected.value = results.poseLandmarks ? true : false
  
  if (results.poseLandmarks) {
    // 计算并发送肩部位置
    const shoulderY = calculateShoulderY(results.poseLandmarks)
    if (shoulderY !== null) {
      // 转换为屏幕坐标
      const screenY = shoulderY * window.innerHeight
      emit('shoulder-position', screenY)  // 发送实际的屏幕坐标
      console.log('Emitting shoulder position:', screenY) // 添加日志
    }
    
    // 绘制骨架
    drawConnectors(ctx, results.poseLandmarks)
    drawLandmarks(ctx, results.poseLandmarks)
  }
}
// 计算肩部中点Y坐标
const calculateShoulderY = (landmarks) => {
  const leftShoulder = landmarks[11]
  const rightShoulder = landmarks[12]
  if (leftShoulder && rightShoulder) {
    const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2
    return shoulderMidY  // 返回 0-1 之间的比例值
  }
  return null
}
// 绘制关键点
const drawLandmarks = (ctx, landmarks) => {
  landmarks.forEach(landmark => {
    const x = landmark.x * ctx.canvas.width
    const y = landmark.y * ctx.canvas.height
    ctx.beginPath()
    ctx.arc(x, y, 5, 0, 2 * Math.PI)
    ctx.fillStyle = '#00FF00'
    ctx.fill()
  })
}
// 绘制连接线
const drawConnectors = (ctx, landmarks) => {
  // 定义需要连接的关键点对
  const connections = [
    [11, 12], // 肩膀
    [11, 13], // 左上臂
    [13, 15], // 左前臂
    [12, 14], // 右上臂
    [14, 16], // 右前臂
    [11, 23], // 左躯干
    [12, 24], // 右躯干
    [23, 24], // 臀部
    [23, 25], // 左大腿
    [25, 27], // 左小腿
    [24, 26], // 右大腿
    [26, 28]  // 右小腿
  ]
  ctx.strokeStyle = '#00FF00'
  ctx.lineWidth = 2
  connections.forEach(([i, j]) => {
    const start = landmarks[i]
    const end = landmarks[j]
    ctx.beginPath()
    ctx.moveTo(start.x * ctx.canvas.width, start.y * ctx.canvas.height)
    ctx.lineTo(end.x * ctx.canvas.width, end.y * ctx.canvas.height)
    ctx.stroke()
  })
}
// 组件挂载时初始化
onMounted(() => {
  initCamera()
})
// 组件卸载时清理
onUnmounted(() => {
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop())
  }
})
</script>
<style scoped>
.camera-container {
  position: fixed;
  left: 20px;
  bottom: 20px;
  width: 20vw;
  height: 30vh;
  background: rgba(0, 0, 0, 0.8);
  border: 2px solid white;
  border-radius: 8px;
  overflow: hidden;
}
.camera-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
}
.camera-video {
  position: absolute;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
}
.pose-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  transform: scaleX(-1);
}
.status-overlay {
  position: absolute;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 10px;
  z-index: 1;
}
.camera-title {
  color: white;
  text-align: center;
  margin: 0;
  background: rgba(0, 0, 0, 0.5);
  padding: 5px;
  border-radius: 4px;
}
.status-text {
  text-align: center;
  padding: 5px;
  color: red;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 4px;
}
.status-text.detected {
  color: #00FF00;
}
</style>
