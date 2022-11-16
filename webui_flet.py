# Flet imports
import flet as ft

# other imports
from math import pi
from typing import Optional
from loguru import logger

class MenuButton(ft.Container):
    def __init__(
        self, title: str, icon: Optional[ft.Control] = None, selected: bool = False
        ):
        super().__init__()
        self.icon = icon
        self.title = title
        self._selected = selected
        self.padding = ft.padding.only(left=43)
        self.height = 38
        self.border_radius = 4
        self.ink = True
        self.on_click = self.item_click

    def item_click(self, _):
        pass

    def _build(self):
        row = ft.Row()
        if self.icon != None:
            row.controls.append(self.icon)
        row.controls.append(ft.Text(self.title))
        self.content = row

    def _before_build_command(self):
        self.bgcolor = "surfacevariant" if self._selected else None
        super()._before_build_command()

class Collapsible(ft.Column):
    def __init__(
        self,
        title: str,
        content: ft.Control,
        icon: Optional[ft.Control] = None,
        spacing: float = 3,
        ):
        super().__init__()
        self.icon = icon
        self.title = title
        self.shevron = ft.Icon(
            ft.icons.KEYBOARD_ARROW_DOWN_ROUNDED,
            animate_rotation=100,
            rotate=0,
        )
        self.content = ft.Column(
            [Container(height=spacing), content],
            height=0,
            spacing=0,
            animate_size=100,
            opacity=0,
            animate_opacity=100,
        )
        self.spacing = 0

    def header_click(self, e):
        self.content.height = None if self.content.height == 0 else 0
        self.content.opacity = 0 if self.content.height == 0 else 1
        self.shevron.rotate = pi if self.shevron.rotate == 0 else 0
        self.update()

    def _build(self):
        title_row = ft.Row()
        if self.icon != None:
            title_row.controls.append(self.icon)
        title_row.controls.append(ft.Text(self.title))
        self.controls.extend(
            [
                Container(
                    ft.Row([title_row, self.shevron], alignment="spaceBetween"),
                    padding=ft.padding.only(left=8, right=8),
                    height=38,
                    border_radius=4,
                    ink=True,
                    on_click=self.header_click,
                    ),
                self.content,
            ]
        )

@logger.catch(reraise=True)
def main(page: ft.Page):
    #def check_item_clicked(e):
        #e.control.checked = not e.control.checked
        #page.update()

    def change_theme(e):
        page.theme_mode = "dark" if page.theme_mode == "light" else "light"

        if "(Light theme)" in theme_switcher.tooltip:
            theme_switcher.tooltip = theme_switcher.tooltip.replace("(Light theme)", '')

        if "(Dark theme)" in theme_switcher.tooltip:
            theme_switcher.tooltip = theme_switcher.tooltip.replace("(Dark theme)", '')

        theme_switcher.tooltip += "(Light theme)" if page.theme_mode == "light" else "(Dark theme)"
        page.update()


    page.title = "Stable Diffusion Playground"
    app_bar_title = ft.Text("Sygil", selectable=True)

    page.theme_mode = "dark"

    theme_switcher = ft.IconButton(ft.icons.WB_SUNNY_OUTLINED, on_click=change_theme,
                                   tooltip=f"Click to change between the light and dark themes. Current {'(Light theme)' if page.theme_mode == 'light' else '(Dark theme)'}")
    
    #
    def close_settings_window(e):
        settings.open = False
        page.update()    
    
    #
    settings = ft.AlertDialog(
            #modal=True,
            title=ft.Text("Settings"),
            content= ft.Row(
                  controls=[
                      ft.Text("Nothing here yet."),
                      ft.Container(
                          width=500,
                          height=500,
                      )
                  ],   
              ),
            actions=[
                # should save options when clicked
                ft.ElevatedButton("Save", icon=ft.icons.SAVE, on_click=close_settings_window),
                
                # Should allow you to discard changed made to the settings.
                ft.ElevatedButton("Discard", icon=ft.icons.RESTORE_FROM_TRASH_ROUNDED, on_click=close_settings_window),
            ],
            actions_alignment="end",
            #on_dismiss=lambda e: print("Modal dialog dismissed!"),
        ) 
    
    def open_settings_window(e):
        page.dialog = settings
        settings.open = True
        page.update()

    menu = ft.PopupMenuButton(
        items=[
                    #ft.PopupMenuItem(text="Settings", on_click=open_settings_modal),
                    ft.PopupMenuItem(),  # divider
                    #ft.PopupMenuItem(text="Checked item", checked=False, on_click=check_item_clicked),
                ])
    #
    
    #
    prompt = ft.TextField(#label="Prompt",
                     value="",
                     min_lines=1,
                     max_lines=1,
                     shift_enter=True,
                     width=1000,
                     #expand=True,
                     tooltip="Prompt to use for generation.",
                     #autofocus=True,
                     hint_text="A corgi wearing a top hat as an oil paiting.",
                     )     
    
    # create the bar at the top
    page.appbar = ft.AppBar(
        #leading=leading,
        #leading_width=leading_width,
        automatically_imply_leading=True,
        title=app_bar_title,
        center_title=False,
        #elevation=5,
        bgcolor=ft.colors.BLACK26,
        actions=[
            ft.Container(width=50),
            prompt,
            ft.ElevatedButton("Generate", on_click=None),
            ft.Container(width=400),
            ft.VerticalDivider(width=10, color="gray"),
            #dropdown,
            theme_switcher,
            ft.IconButton(icon=ft.icons.SETTINGS, on_click=open_settings_window),
            menu,
            ],
    )  

    # create a colum for a sidebar on the left side of the page
    left_col = ft.Column(
        width=50,
        controls=[
            # Create a container so we can group buttons, change bgcolor and drag it around later.
            # add some buttons inside containers so we can rearrange them if needed and drop things in them.
            #ft.Container(ft.IconButton(icon=ft.icons.MENU_OUTLINED, tooltip='')),
            
            ft.Container(ft.IconButton(width=50, content = ft.Icon(ft.icons.ADD_OUTLINED), tooltip ='Import Image', on_click = None, disabled=True)),
            ft.Container(ft.IconButton(width=50, content = ft.Icon(ft.icons.DASHBOARD_OUTLINED), tooltip = 'Gallery', on_click = None, disabled=True)),

            ft.Draggable(content=ft.Divider(height=10, color="white")),
            ],
    )
                        #]
                    #)
                    
    #
    # Canvas container
    canvas = ft.Container(content=
                 ft.Stack(
                     [
                         #ft.Row([
                             #ft.Image(
                                 #src=f"https://static.wixstatic.com/media/ac4dba_13be94c39c804e8aa2131a51036a0244~mv2.png",
                                 #width=300,
                                 #height=30,
                                 #fit="contain",
                                 #expand=True,
                                 #), 
                             #]),
                         ft.Stack([
                             ft.Image(
                                 src=f"https://i.redd.it/qdxksbar05o31.jpg",
                                 #width=300,
                                 #height=300,
                                 #fit="contain",
                                 gapless_playback=True,
                                 expand=True,
                                 ),    
                             ],
                             ),
                        ],
                     #width=40,
                     #height=40,
                     clip_behavior=None,
                     ),
                 alignment=ft.alignment.center, #type: ignore
                 bgcolor=ft.colors.WHITE10,

                 expand=True
                )    

    
    
    # options
    options = ft.Container(content=
                           ft.Stack( controls=[
                               ft.Row([  
                                   ft.Dropdown(label="Custom Models", options=[
                                       ft.dropdown.Option("Stable Diffusion 1.5"),
                                       ft.dropdown.Option("Waifu Diffusion 1.3"),
                                       ft.dropdown.Option("MM-27 Merged Pruned"),
                                       ],
                                        height=70,
                                        width=210,
                                        value="Stable Diffusion 1.5",
                                        tooltip="Custom models located in your `models/custom` folder including the default stable diffusion model.",
                                        ),
                         
                         ft.TextField(label="Sampling Steps", value=30, height=70, width=170,tooltip="Sampling steps.", keyboard_type="number"),
                         
                         ft.TextField(label="Width", value=512, height=50, width=100, suffix_text="W",tooltip="Widgth in pixels.", keyboard_type="number"),
                         
                         ft.TextField(label="Height", value=512, height=50, width=100, suffix_text="H", tooltip="Height in pixels.",keyboard_type="number"),
                         
                         ft.TextField(label="CFG", value=7.5, height=50, width=100, #suffix_text="CFG",
                                      tooltip="Classifier Free Guidance Scale.", keyboard_type="number"),
                         
                         ft.TextField(label="Seed", value=-1, height=60, width=210, #suffix_text="seed", 
                                      tooltip="Seed used for the generation, leave empty or use -1 for a random seed. You can also use word as seeds.",
                                      keyboard_type="number"),
                         
                         ft.Dropdown(label="Sampling method", options=[ #["k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
                             ft.dropdown.Option("k_lms"),
                             ft.dropdown.Option("k_euler"),
                             ft.dropdown.Option("k_euler_a"),
                             ft.dropdown.Option("k_dpm_2"),
                             ft.dropdown.Option("k_dpm_2_a"),
                             ft.dropdown.Option("k_heun"),
                             ft.dropdown.Option("PLMS"),
                             ft.dropdown.Option("DDIM"),
                             ],
                              height=60,
                              width=150,
                              value="k_lms",
                              tooltip="Sampling method or scheduler to use, different sampling method"
                              " or schedulers behave differently giving better or worst performance in more or less steps."
                              "Try to find the best one for your needs and hardware.",
                              ),
                         
                         ft.Switch(label="Show Batch Options", value=False, disabled=True, tooltip="Option disabled for now."),
                         ft.Switch(label="Preview Image Settings", value=False, disabled=True, tooltip="Option disabled for now."),
                         
                         ],
                        wrap=True,
                        spacing=4,
                        ),
                        ]  
                        ),
                 expand=True)  

    # create a colum for a panel on the right side of the page
    right_col = ft.Column(
        width=400,
        controls=[
            ft.Draggable(content=ft.Divider(height=10, color="gray")),
            
            # prompt
            #prompt,
            
            options,
            
            ft.Draggable(content=ft.Divider(height=10, color="gray")),
            ft.Container(#height=380,
                         expand=True),
            ft.Draggable(content=ft.Divider(height=10, color="gray")),
            ]
        )

    page.add(
        ft.Row(
            [
                # left side menu and toolbar
                left_col,
                ft.Draggable(content=ft.VerticalDivider(width=10, color="gray")),

                # canvas 
                canvas,
                
                # resizable right side panel with options and tools.
                ft.Draggable(content=ft.VerticalDivider(width=10, color="gray")),
                right_col,
                ],
            expand=True,
            ),
    )

    # bottom panel 
    bottom_panel = ft.Container(content=
                                ft.Stack( [ ft.Tooltip( message="Nothing to see here as this panel is not yet implemented.",
                                                        content=ft.Container(bgcolor=ft.colors.BLACK12, height=200)) ] ))     
    
    #page.add(ft.Container(ft.Text("test", selectable=True),height=500))
    
    page.add(bottom_panel)

ft.app(target=main, port=8505)