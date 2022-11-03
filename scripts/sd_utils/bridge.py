# This file is part of sygil-webui (https://github.com/Sygil-Dev/sygil-webui/).

# Copyright 2022 Sygil-Dev team.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# base webui import and utils.
#import streamlit as st

# We import hydralit like this to replace the previous stuff
# we had with native streamlit as it lets ur replace things 1:1
from sd_utils import logger, load_models

# streamlit imports

#streamlit components section

#other imports
import requests, time, json, base64
from io import BytesIO

# import custom components


# end of imports
#---------------------------------------------------------------------------------------------------------------



@logger.catch(reraise=True)
def run_bridge(interval, api_key, horde_name, horde_url, priority_usernames, horde_max_pixels, horde_nsfw, horde_censor_nsfw, horde_blacklist, horde_censorlist):
    current_id = None
    current_payload = None
    loop_retry = 0
    # load the model for stable horde if its not in memory already
    # we should load it after we get the request from the API in
    # case the model is different from the loaded in memory but
    # for now we can load it here so its read right away.
    load_models(use_GFPGAN=True)
    while True:

        if loop_retry > 10 and current_id:
            logger.info(f"Exceeded retry count {loop_retry} for generation id {current_id}. Aborting generation!")
            current_id = None
            current_payload = None
            current_generation = None
            loop_retry = 0
        elif current_id:
            logger.info(f"Retrying ({loop_retry}/10) for generation id {current_id}...")
        gen_dict = {
            "name": horde_name,
            "max_pixels": horde_max_pixels,
            "priority_usernames": priority_usernames,
            "nsfw": horde_nsfw,
            "blacklist": horde_blacklist,
            "models": ["stable_diffusion"],
        }
        headers = {"apikey": api_key}
        if current_id:
            loop_retry += 1
        else:
            try:
                pop_req = requests.post(horde_url + '/api/v2/generate/pop', json = gen_dict, headers = headers)
            except requests.exceptions.ConnectionError:
                logger.warning(f"Server {horde_url} unavailable during pop. Waiting 10 seconds...")
                time.sleep(10)
                continue
            except requests.exceptions.JSONDecodeError():
                logger.warning(f"Server {horde_url} unavailable during pop. Waiting 10 seconds...")
                time.sleep(10)
                continue
            try:
                pop = pop_req.json()
            except json.decoder.JSONDecodeError:
                logger.warning(f"Could not decode response from {horde_url} as json. Please inform its administrator!")
                time.sleep(interval)
                continue
            if pop == None:
                logger.warning(f"Something has gone wrong with {horde_url}. Please inform its administrator!")
                time.sleep(interval)
                continue
            if not pop_req.ok:
                message = pop['message']
                logger.warning(f"During gen pop, server {horde_url} responded with status code {pop_req.status_code}: {pop['message']}. Waiting for 10 seconds...")
                if 'errors' in pop:
                    logger.debug(f"Detailed Request Errors: {pop['errors']}")
                time.sleep(10)
                continue
            if not pop.get("id"):
                skipped_info = pop.get('skipped')
                if skipped_info and len(skipped_info):
                    skipped_info = f" Skipped Info: {skipped_info}."
                else:
                    skipped_info = ''
                logger.info(f"Server {horde_url} has no valid generations to do for us.{skipped_info}")
                time.sleep(interval)
                continue
            current_id = pop['id']
            logger.info(f"Request with id {current_id} picked up. Initiating work...")
            current_payload = pop['payload']
            if 'toggles' in current_payload and current_payload['toggles'] == None:
                logger.error(f"Received Bad payload: {pop}")
                current_id = None
                current_payload = None
                current_generation = None
                loop_retry = 0
                time.sleep(10)
                continue

        logger.debug(current_payload)
        current_payload['toggles'] = current_payload.get('toggles', [1,4])
        # In bridge-mode, matrix is prepared on the horde and split in multiple nodes
        if 0 in current_payload['toggles']:
            current_payload['toggles'].remove(0)
        if 8 not in current_payload['toggles']:
            if horde_censor_nsfw and not horde_nsfw:
                current_payload['toggles'].append(8)
            elif any(word in current_payload['prompt'] for word in horde_censorlist):
                current_payload['toggles'].append(8)

        from txt2img import txt2img


        """{'prompt': 'Centred Husky, inside spiral with circular patterns, trending on dribbble, knotwork, spirals, key patterns,
        zoomorphics, ', 'ddim_steps': 30, 'n_iter': 1, 'sampler_name': 'DDIM', 'cfg_scale': 16.0, 'seed': '3405278433', 'height': 512, 'width': 512}"""

        #images, seed, info, stats = txt2img(**current_payload)
        images, seed, info, stats = txt2img(str(current_payload['prompt']), int(current_payload['ddim_steps']), str(current_payload['sampler_name']),
                                                    int(current_payload['n_iter']), 1, float(current_payload["cfg_scale"]), str(current_payload["seed"]),
                                                    int(current_payload["height"]), int(current_payload["width"]), save_grid=False, group_by_prompt=False,
                                                    save_individual_images=False,write_info_files=False)

        buffer = BytesIO()
        # We send as WebP to avoid using all the horde bandwidth
        images[0].save(buffer, format="WebP", quality=90)
        # logger.info(info)
        submit_dict = {
            "id": current_id,
            "generation": base64.b64encode(buffer.getvalue()).decode("utf8"),
            "api_key": api_key,
            "seed": seed,
            "max_pixels": horde_max_pixels,
        }
        current_generation = seed
        while current_id and current_generation != None:
            try:
                submit_req = requests.post(horde_url + '/api/v2/generate/submit', json = submit_dict, headers = headers)
                try:
                    submit = submit_req.json()
                except json.decoder.JSONDecodeError:
                    logger.error(f"Something has gone wrong with {horde_url} during submit. Please inform its administrator!  (Retry {loop_retry}/10)")
                    time.sleep(interval)
                    continue
                if submit_req.status_code == 404:
                    logger.info(f"The generation we were working on got stale. Aborting!")
                elif not submit_req.ok:
                    logger.error(f"During gen submit, server {horde_url} responded with status code {submit_req.status_code}: {submit['message']}. Waiting for 10 seconds...  (Retry {loop_retry}/10)")
                    if 'errors' in submit:
                        logger.debug(f"Detailed Request Errors: {submit['errors']}")
                    time.sleep(10)
                    continue
                else:
                    logger.info(f'Submitted generation with id {current_id} and contributed for {submit_req.json()["reward"]}')
                current_id = None
                current_payload = None
                current_generation = None
                loop_retry = 0
            except requests.exceptions.ConnectionError:
                logger.warning(f"Server {horde_url} unavailable during submit. Waiting 10 seconds...  (Retry {loop_retry}/10)")
                time.sleep(10)
                continue
        time.sleep(interval)
