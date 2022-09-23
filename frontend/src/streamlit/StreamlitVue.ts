/**
 * Vue.js specific composables
 */
import { onMounted, onUpdated } from "vue"
import { Streamlit } from "streamlit-component-lib"

export function useStreamlit() {
  /**
   * Optional Streamlit Vue-based setup.
   *
   * You are not required call this function on your Streamlit
   * component. If you decide not to call it, you should implement the
   * `onMounted` and `onUpdated` functions in your own component,
   * so that your plugin properly resizes.
   */

  onMounted((): void => {
    // After we're rendered for the first time, tell Streamlit that our height
    // has changed.
    Streamlit.setFrameHeight()
  })

  onUpdated((): void => {
    // After we're updated, tell Streamlit that our height may have changed.
    Streamlit.setFrameHeight()
  })
}
