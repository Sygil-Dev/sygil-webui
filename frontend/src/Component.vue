<template>
  <div class="bootstrap-wrapper pt-4">
    <div class="concept-gallery row">
      <div v-for="concept in args.concepts"
           :key="concept.name"
           class="col-12 col-sm-6 col-md-4 col-lg-4 col-xl-3">

        <div class="concept-card p-4 container-fluid">
          <div class="concept-card-content-wrapper">
            <div class="card-header row no-gutters">
              <div class="col">
                <h1 class="concept-title pl-1"><span class="token-char pr-0">&lt;</span>{{ concept.name }}<span class="pl-0 token-char">&gt;</span></h1>
              </div>

              <!-- Favorite feature, not implemented yet -->
              <!-- <div class="col-auto card-favorite" >
              <img width="24"
                   height="24"
                   class="icon-star"
                   src="./icons/star.svg" />
              </div>  -->

            </div>

            <div class="concept-img-wrapper p-0 row no-gutters">

              <div v-for="(img, img_index) in concept.images"
                   :key="'concept_img'+img_index"
                   :class="{
                     'p-1': true,
                      'col-6': concept.images.length % 2 == 0 || img_index < concept.images.length - 1,
                      'col-12': concept.images.length % 2 == 1 && img_index == concept.images.length - 1
                   }">
                   <div class="img-bg" :style="{'background-image': 'url(data:image/png;base64,' + img + ')'}"></div>

              </div>

              <div v-if="concept.images.length == 0"
                   class="col-12 p-4 no-preview">
                   <svg class="no-preview-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M 2 5 L 2 27 L 30 27 L 30 5 Z M 4 7 L 28 7 L 28 20.90625 L 22.71875 15.59375 L 22 14.875 L 17.46875 19.40625 L 11.71875 13.59375 L 11 12.875 L 4 19.875 Z M 24 9 C 22.894531 9 22 9.894531 22 11 C 22 12.105469 22.894531 13 24 13 C 25.105469 13 26 12.105469 26 11 C 26 9.894531 25.105469 9 24 9 Z M 11 15.71875 L 20.1875 25 L 4 25 L 4 22.71875 Z M 22 17.71875 L 28 23.71875 L 28 25 L 23.03125 25 L 18.875 20.8125 Z"/></svg>
                 <p style="opacity: 0.8">No preview available</p>
              </div>

            </div>

            <div class="concept-card-footer row no-gutters pt-4">
              <div class="col pl-1">
                <div v-if="concept.type"
                     :class="{

                      'concept-type-tag': true,
                      'concept-type-style': concept.type.toLowerCase() === 'style',
                      'concept-type-object': concept.type.toLowerCase() === 'object'
                    }
                     ">
                  {{ concept.type.toUpperCase() }}
                </div>
              </div>
              <div class="col-auto">
                <!-- Copy to clipboard button -->
                <button class="button"
                        @click="copyToClipboard(concept.token)">
                        <!-- <svg class="icon-clipboard" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" ><path d="M 15 3 C 13.742188 3 12.847656 3.890625 12.40625 5 L 5 5 L 5 28 L 13 28 L 13 30 L 27 30 L 27 14 L 25 14 L 25 5 L 17.59375 5 C 17.152344 3.890625 16.257813 3 15 3 Z M 15 5 C 15.554688 5 16 5.445313 16 6 L 16 7 L 19 7 L 19 9 L 11 9 L 11 7 L 14 7 L 14 6 C 14 5.445313 14.445313 5 15 5 Z M 7 7 L 9 7 L 9 11 L 21 11 L 21 7 L 23 7 L 23 14 L 13 14 L 13 26 L 7 26 Z M 15 16 L 25 16 L 25 28 L 15 28 Z"/></svg> -->
                  Copy to clipboard
                </button>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue"
import { Streamlit, Theme } from "streamlit-component-lib"
import { useStreamlit } from "./streamlit"
interface IProps {
  args: any;
  disabled: boolean;
  theme: Theme;
}

useStreamlit(); // lifecycle hooks for automatic Streamlit resize
const props = defineProps<IProps>();

const copyToClipboard = (text: string) => {
  console.log("sending copy to clipboard event", text)
  // Streamlit.setComponentValue({
  //   action: "copy_to_clipboard",
  //   value: text
  // })
  // copy to clipboard
  navigator.clipboard.writeText(text)
}

// const enable_favorites = ref(false)
// const enable_copy_to_clipboard = ref(false)


</script>
<style>

/* svg.icon-clipboard  {
    fill: var(--text-color);
    width: 18px;
    height: 18px;
} */

svg.no-preview-icon {
    fill: var(--text-color);
    width: 72px;
    height: 72px;
    opacity: 0.2;

}

.no-preview {
  align-self: center;
    text-align: center;
    color: var(--text-color);
}
.concept-card {
  background-color: var(--secondary-background-color);
  border-radius: 5px;
  margin-bottom: 20px;

}

.concept-card-content-wrapper {
  flex-direction: column !important;
  display: flex !important;
  height: 360px;
}

.concept-title {
  margin-top: 0px;
  margin-bottom: 24px;
  font-size: 1em;
  color: var(--text-color);
}

.concept-img-wrapper {
  flex-grow: 1 !important;

}

.card-favorite {
  text-align: end;
}

.concept-img {
  max-height: 100%;
  height: 100%;
}

.concept-img img {
  border-radius: 8px;
  object-fit: cover;
}

.img-bg {
  background-size: cover;
  background-position: center;
  background-origin: content-box;
  background-repeat: no-repeat;
  height: 100%;
  width: 100%;
  border-radius: 8px;
}

.icon-star {
  cursor: pointer;
  position: relative;
  top: -3px;
}

.token-char {
  color: #939393;
  font-weight: 700;
  position: relative;
  top: 1px;
}

.concept-card-footer {
  align-items: center;
}

.concept-type-tag {
  background-color: #898989;
  border-radius: 16px;
  padding: 5px 16px;
  font-size: 0.7em;
  color: #fff;
  display: inline-block;
  font-weight: bold;
}

.concept-type-style {
  background-color: #0095ff;
}

.concept-type-object {
  background-color: #ff9031;
}

.button {
  height: 35px;
  cursor: pointer;
  display: inline-flex;
  -webkit-box-align: center;
  align-items: center;
  -webkit-box-pack: center;
  justify-content: center;
  font-weight: 400;
  padding: 0.25rem 0.75rem;
  border-radius: 0.25rem;
  margin: 0px;
  line-height: 1.6;
  color: inherit;
  width: auto;
  user-select: none;
  background-color: var(--background-color);
  border: 1px solid rgba(128, 128, 128, 0.8);
}

.button:hover {
  color: var(--primary-color);
  border-color: var(--primary-color);
}

/* .button:focus {
  box-shadow: rgb(var(--primary-color) / 50%) 0px 0px 0px 0.2rem;
  outline: none;
}

.button:focus:not(:active) {
    border-color: var(--primary-color);
    color: var(--primary-color);
} */
</style>