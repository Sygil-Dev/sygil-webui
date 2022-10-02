window.SD = (() => {
  /*
   * Painterro is made a field of the SD global object
   * To provide convinience when using w() method in css_and_js.py
   */
  class PainterroClass {
    static isOpen = false;
    static async init ({ x, toId }) {
      console.log(x)

      const originalImage = x[2] === 'Mask' ? x[1]?.image : x[0];

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
    static success (result) { return [result, { image: result, mask: result }] };
    static fallback (image) { return [image, { image: image, mask: image }] };
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
      x = x[0];
      if (!Array.isArray(x) || x.length === 0) return;

      this.clearImageInput(this.el.get(`#${toId}`));

      const i = this.#getGallerySelectedIndex(this.el.get(`#${fromId}`));

      return [x[i].replace('data:;','data:image/png;')];
    }
    async copyImageFromGalleryToClipboard ({ x, fromId }) {
      x = x[0];
      if (!Array.isArray(x) || x.length === 0) return;

      const i = this.#getGallerySelectedIndex(this.el.get(`#${fromId}`));

      const data = x[i];
      const blob = await (await fetch(data.replace('data:;','data:image/png;'))).blob();
      const item = new ClipboardItem({'image/png': blob});

      await this.copyToClipboard([item]);
    }
    async copyFullOutput ({ fromId }) {
      const textField = this.el.get(`#${fromId} .textfield`);
      if (!textField) {
        SDclass.error(new Error(`Can't find textfield with the output!`));
      }

      const value = textField.textContent.replace(/\s+/g,' ').replace(/: /g,':');

      await this.copyToClipboard(value)
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
    async gradioInputToClipboard ({ x }) { return this.copyToClipboard(x[0]); }
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
