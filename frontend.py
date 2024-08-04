import easygui
import kivy
import random
from kivy.app import runTouchApp
from kivy.lang import Builder
from kivy.app import App
from kivy.graphics import Color, Rectangle
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from app import image_method
from kivy.uix.textinput import TextInput


#from kivy.config import Config


class RootWidget(GridLayout):
    pass


class MainApp(App):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'convert handwritten text to typed text'
        self.file_name = ""
        self.text_date = ""

    def build(self):
        self.xx = ""
        root = RootWidget(cols=2)
        image_source = Image(id='image_source', source='', size_hint=(0.5, 0.9), pos_hint={'x': 0, 'top': 0}, opacity=0)
        root.add_widget(image_source)
        text_output = Label(id='text_output', text='', size_hint=(0.5, 0.9), pos_hint={'right': 1, 'y': 0})
        root.add_widget(text_output)
        btn = Button(id='image_btn', text="choose image to convert", background_color=[0.2, 0.2, 1, 1],
                     size_hint=(0.5, 0.1), pos_hint={'x': 0, 'y': 1})
        self.xx = btn.bind(on_press=self.on_button_press)
        root.add_widget(btn)
        btn2 = Button(id='convert_btn', text="convert", background_color=[0.2, 0.2, 1, 1], size_hint=(0.5, 0.1),
                      pos_hint={'right': 1, 'top': 1})
        btn2.bind(on_press=self.on_button_press)
        root.add_widget(btn2)

        return root

    def on_button_press(self, button):
        for widget in button.parent.walk(loopback=True):
            if widget.id == 'image_source' and button.id == 'image_btn':
                self.file_name = easygui.fileopenbox(filetypes=['*.jpg', '*.png', '*.gif'])
                widget.source = self.file_name
                widget.allow_stretch = True
                widget.opacity = 1

            elif widget.id == 'text_output' and button.id == 'convert_btn':
                widget.text = image_method(self.file_name)

        return


if __name__ == "__main__":
    app = MainApp()
    app.run()
