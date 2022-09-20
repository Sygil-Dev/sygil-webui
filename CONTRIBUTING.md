# Contribution Guide

All Pull Requests are opened against `dev` branch which is our main development branch.

There are two UI systems that are supported currently:

* **Gradio** — entry point is in the `/scripts/webui.py` you can start from there. Check out [Gradio documentation](https://gradio.app/docs/) and their [Discord channel](https://discord.gg/Qs8AsnX7Jd) for more information about Gradio.
* **Streamlit** — entry point is in the `/scripts/webui_streamlit.py`. Documentation on Streamlit is [located here](https://docs.streamlit.io/).

### Development environment

`environment.yaml` can be different from the one on `master` so be sure to update before making any changes to the code.

The development environment is currently very similar to the one in production, so you can work on your contribution in the same conda env. Optionally you can create a separate environment.

### Making changes

If you're working on a fix please post about it in the respective issue. If the issue doesn't exist create it and then mention it in your Pull Request.

If you're introducing new features please make the corresponding additions to the documentation with an explanation of the new behavior. The documentation is located in `/docs/`. Depending on your contribution you may edit the existing files in there or create a new one.

### Opening a Pull Request

Prior to opening a request make sure your Web UI works locally with your changes and that your branch is up-to-date with the main repository. Finally, open a new PR against `dev` branch.
