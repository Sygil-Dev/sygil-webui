<template>
  <div>
    <!--  Error boundary. If our wrapped component threw an error, display it. -->
    <div v-if="componentError !== ''">
      <h1 class="err__title">Component Error</h1>
      <div class="err__msg">Message: {{ componentError }}</div>
    </div>
    <!-- 
      Else render the component slot and pass Streamlit event data in `args` props to it.
      Don't render until we've gotten our first RENDER_EVENT from Streamlit.
      All components get disabled while the app is being re-run, and become re-enabled when the re-run has finished.
    -->
    <slot
      v-else-if="renderData != null"
      :args="renderData.args"
      :theme="renderData.theme"
      :disabled="renderData.disabled"
    ></slot>
  </div>
</template>

<script lang="ts">
import {
  ref,
  defineComponent,
  onMounted,
  onUpdated,
  onUnmounted,
  onErrorCaptured,
} from "vue"
import { Streamlit, RenderData } from "streamlit-component-lib"

export default defineComponent({
  name: "WithStreamlitConnection",
  setup() {
    const renderData = ref<RenderData>((undefined as unknown) as RenderData)
    const componentError = ref("")

    const onRenderEvent = (event: Event): void => {
      const renderEvent = event as CustomEvent<RenderData>
      renderData.value = renderEvent.detail
      componentError.value = ""
    }

    // Set up event listeners, and signal to Streamlit that we're ready.
    // We won't render the component until we receive the first RENDER_EVENT.
    onMounted(() => {
      Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRenderEvent)
      Streamlit.setComponentReady()
    })
    onUpdated(() => {
      // If our slot threw an error, we display it in render(). In this
      // case, the slot won't be mounted and therefore won't call
      // `setFrameHeight` on its own. We do it here so that the rendered
      // error will be visible.
      if (componentError.value != "") {
        Streamlit.setFrameHeight()
      }
    })
    onUnmounted(() => {
      Streamlit.events.removeEventListener(
        Streamlit.RENDER_EVENT,
        onRenderEvent
      )
    })
    onErrorCaptured(err => {
      componentError.value = String(err)
    })

    return {
      renderData,
      componentError,
    }
  },
})
</script>

<style scoped>
.err__title,
.err__msg {
  margin: 0;
}
</style>
