window.setTimeout(() => {
    let input = document.querySelector('gradio-app').shadowRoot.querySelector('#prompt_input [data-testid="textbox"]')
    console.log(input)
    input.addEventListener('keyup', () => {
      console.log(input.value)
      const prompt_input = document.querySelector('gradio-app').shadowRoot.querySelector('#prompt_input [data-testid="textbox"]');
      const multiline = document.querySelector('gradio-app').shadowRoot.querySelector('#submit_on_enter label:nth-child(2)')
      if (prompt_input.scrollWidth > prompt_input.clientWidth + 10 ) {
         multiline.click();
      }

      txt2img_prompt = input.value
      let height_match = /(?:-h|-H|--height|height)[ :]?(?<height>\d+) /.exec(txt2img_prompt);
      let txt2img_width, txt2img_height, txt2img_steps, txt2img_seed, txt2img_batch_count, txt2img_cfg;
      if (height_match) {
        txt2img_height = Math.round(height_match.groups.height / 64) * 64;
        txt2img_prompt = txt2img_prompt.replace(height_match[0], '');
      }
      let width_match = /(?:-w|-W|--width|width)[ :]?(?<width>\d+) /.exec(txt2img_prompt);
      if (width_match) {
        txt2img_width = Math.round(width_match.groups.width / 64) * 64;
        txt2img_prompt = txt2img_prompt.replace(width_match[0], '');
      }
      let steps_match = /(?:-s|--steps|steps)[ :]?(?<steps>\d+) /.exec(txt2img_prompt);
      if (steps_match) {
        txt2img_steps = steps_match.groups.steps.trim();
        txt2img_prompt = txt2img_prompt.replace(steps_match[0], '');
      }
      let seed_match = /(?:-S|--seed|seed)[ :]?(?<seed>\d+) /.exec(txt2img_prompt);
      if (seed_match) {
        txt2img_seed = seed_match.groups.seed;
        txt2img_prompt = txt2img_prompt.replace(seed_match[0], '');
      }
      let batch_count_match = /(?:-n|-N|--number|number)[ :]?(?<batch_count>\d+) /.exec(txt2img_prompt);
      if (batch_count_match) {
        txt2img_batch_count = batch_count_match.groups.batch_count;
        txt2img_prompt = txt2img_prompt.replace(batch_count_match[0], '');
      }
      let cfg_scale_match = /(?:-c|-C|--cfg-scale|cfg_scale|cfg)[ :]?(?<cfgscale>\d\.?\d+?) /.exec(txt2img_prompt);
      if (cfg_scale_match) {
        txt2img_cfg = parseFloat(cfg_scale_match.groups.cfgscale).toFixed(1);
        txt2img_prompt = txt2img_prompt.replace(cfg_scale_match[0], '');
      }
      let sampler_match = /(?:-A|--sampler|sampler)[ :]?(?<sampler>\w+) /.exec(txt2img_prompt);
      if (sampler_match) {

        txt2img_prompt = txt2img_prompt.replace(sampler_match[0], '');
      }
      input.value = txt2img_prompt;
      console.log(input.value, parseInt(txt2img_width), parseInt(txt2img_height), parseInt(txt2img_steps), txt2img_seed, parseInt(txt2img_batch_count), parseFloat(txt2img_cfg))

    })

   // function debounce(func, timeout = 300){
   //      let timer;
   //      return (...args) => {
   //        clearTimeout(timer);
   //        timer = setTimeout(() => { func.apply(this, args); }, timeout);
   //      };
   //    }
   //    function saveInput(){
   //      console.log('Saving data');
   //    }
   //  const processChange = debounce(() => saveInput());

}, 500)


window.SD = (() => {


  /*
   * Painterro is made a field of the SD global object
   * To provide convinience when using w() method in css_and_js.py
   */
  class PainterroClass {
    static isOpen = false;
    static async init ({ x, toId }) {
      const img = x;
      const originalImage = Array.isArray(img) ? img[0] : img;

      if (window.Painterro === undefined) {
        try {
          await this.load();
        } catch (e) {
          SDClass.error(e);

          return this.fallback(originalImage);
        }
      }

      if (this.isOpen) {
        return this.fallback(originalImage);
      }
      this.isOpen = true;

      let resolveResult;
      const paintClient = Painterro({
        hiddenTools: ['arrow'],
        onHide: () => {
          resolveResult?.(null);
        },
        saveHandler: (image, done) => {
          const data = image.asDataURL();

          // ensures stable performance even
          // when the editor is in interactive mode
          SD.clearImageInput(SD.el.get(`#${toId}`));

          resolveResult(data);

          done(true);
          paintClient.hide();
        },
      });

      const result = await new Promise((resolve) => {
        resolveResult = resolve;
        paintClient.show(originalImage);
      });
      this.isOpen = false;

      return result ? this.success(result) : this.fallback(originalImage);
    }

    static success (result) { return [result, result]; }

    static fallback (image) { return [image, image]; }

    static load () {
      return new Promise((resolve, reject) => {
        const scriptId = '__painterro-script';
        if (document.getElementById(scriptId)) {
          reject(new Error('Tried to load painterro script, but script tag already exists.'));
          return;
        }

        const styleId = '__painterro-css-override';
        if (!document.getElementById(styleId)) {
          /* Ensure Painterro window is always on top */
          const style = document.createElement('style');
          style.id = styleId;
          style.setAttribute('type', 'text/css');
          style.appendChild(document.createTextNode(`
            .ptro-holder-wrapper {
                z-index: 100;
            }
          `));
          document.head.appendChild(style);
        }

        const script = document.createElement('script');
        script.id = scriptId;
        script.src = 'https://unpkg.com/painterro@1.2.78/build/painterro.min.js';
        script.onload = () => resolve(true);
        script.onerror = (e) => {
          // remove self on error to enable reattempting load
          document.head.removeChild(script);
          reject(e);
        };
        document.head.appendChild(script);
      });
    }
  }

  /*
   * Turns out caching elements doesn't actually work in gradio
   * As elements in tabs might get recreated
   */
  class ElementCache {
    #el;
    constructor () {
      this.root = document.querySelector('gradio-app').shadowRoot;
    }
    get (selector) {
      return this.root.querySelector(selector);
    }
  }

  /*
   * The main helper class to incapsulate functions
   * that change gradio ui functionality
   */
  class SDClass {
    el = new ElementCache();
    Painterro = PainterroClass;

    moveImageFromGallery ({ x, fromId, toId }) {
      if (!Array.isArray(x) || x.length === 0) return;

      this.clearImageInput(this.el.get(`#${toId}`));

      const i = this.#getGallerySelectedIndex(this.el.get(`#${fromId}`));

      return [x[i].replace('data:;','data:image/png;')];
    }




    async copyImageFromGalleryToClipboard ({ x, fromId }) {
      if (!Array.isArray(x) || x.length === 0) return;

      const i = this.#getGallerySelectedIndex(this.el.get(`#${fromId}`));

      const data = x[i];
      const blob = await (await fetch(data.replace('data:;','data:image/png;'))).blob();
      const item = new ClipboardItem({'image/png': blob});

      await this.copyToClipboard([item]);
    }

    clickFirstVisibleButton({ rowId }) {
      const generateButtons = this.el.get(`#${rowId}`).querySelectorAll('.gr-button-primary');

      if (!generateButtons) return;

      for (let i = 0, arr = [...generateButtons]; i < arr.length; i++) {
        const cs = window.getComputedStyle(arr[i]);

        if (cs.display !== 'none' && cs.visibility !== 'hidden') {
          console.log(arr[i]);

          arr[i].click();
          break;
        }
      }
    }

    async gradioInputToClipboard ({ x }) { return this.copyToClipboard(x); }

    async copyToClipboard (value) {
      if (!value || typeof value === 'boolean') return;
      try {
        if (Array.isArray(value) &&
            value.length &&
            value[0] instanceof ClipboardItem) {
          await navigator.clipboard.write(value);
        } else {
          await navigator.clipboard.writeText(value);
        }
      } catch (e) {
        SDClass.error(e);
      }
    }

    static error (e) {
      console.error(e);
      if (typeof e === 'string') {
        alert(e);
      } else if(typeof e === 'object' && Object.hasOwn(e, 'message')) {
        alert(e.message);
      }
    }

    clearImageInput (imageEditor) {
      imageEditor?.querySelector('.modify-upload button:last-child')?.click();
    }

    #getGallerySelectedIndex (gallery) {
      const selected = gallery.querySelector(`.\\!ring-2`);
      return selected ? [...selected.parentNode.children].indexOf(selected) : 0;
    }
  }

  return new SDClass();
})();
