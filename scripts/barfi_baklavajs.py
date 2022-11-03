# This file is part of sygil-webui (https://github.com/Sygil-Dev/sandbox-webui/).

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
#from sd_utils import *
from sd_utils import *
# streamlit imports

#streamlit components section

#other imports
import os, time, requests
import sys

from barfi import st_barfi, barfi_schemas, Block

# Temp imports

# end of imports
#---------------------------------------------------------------------------------------------------------------


def layout():
    #st.info("Under Construction. :construction_worker:")

    #from barfi import st_barfi, Block

    #add = Block(name='Addition')
    #sub = Block(name='Subtraction')
    #mul = Block(name='Multiplication')
    #div = Block(name='Division')

    #barfi_result = st_barfi(base_blocks= [add, sub, mul, div])
    # or if you want to use a category to organise them in the frontend sub-menu
    #barfi_result = st_barfi(base_blocks= {'Op 1': [add, sub], 'Op 2': [mul, div]})

    col1, col2, col3 = st.columns([1, 8, 1])

    from barfi import st_barfi, barfi_schemas, Block


    with col2:
        feed = Block(name='Feed')
        feed.add_output()
        def feed_func(self):
            self.set_interface(name='Output 1', value=4)
        feed.add_compute(feed_func)

        splitter = Block(name='Splitter')
        splitter.add_input()
        splitter.add_output()
        splitter.add_output()
        def splitter_func(self):
            in_1 = self.get_interface(name='Input 1')
            value = (in_1/2)
            self.set_interface(name='Output 1', value=value)
            self.set_interface(name='Output 2', value=value)
        splitter.add_compute(splitter_func)

        mixer = Block(name='Mixer')
        mixer.add_input()
        mixer.add_input()
        mixer.add_output()
        def mixer_func(self):
            in_1 = self.get_interface(name='Input 1')
            in_2 = self.get_interface(name='Input 2')
            value = (in_1 + in_2)
            self.set_interface(name='Output 1', value=value)
        mixer.add_compute(mixer_func)

        result = Block(name='Result')
        result.add_input()
        def result_func(self):
            in_1 = self.get_interface(name='Input 1')
        result.add_compute(result_func)

        load_schema = st.selectbox('Select a saved schema:', barfi_schemas())

        compute_engine = st.checkbox('Activate barfi compute engine', value=False)

        barfi_result = st_barfi(base_blocks=[feed, result, mixer, splitter],
                            compute_engine=compute_engine, load_schema=load_schema)

        if barfi_result:
            st.write(barfi_result)
