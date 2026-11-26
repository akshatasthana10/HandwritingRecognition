import os

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup

from app import image_method

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


class MainApp(App):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'convert handwritten text to typed text'
        self.file_name = ""

    def build(self):
        root = GridLayout(cols=2)
        self.image_source = Image(source='', size_hint=(0.5, 0.9), fit_mode="contain", opacity=0)
        root.add_widget(self.image_source)
        self.text_output = Label(text='', size_hint=(0.5, 0.9), halign='left', valign='middle')
        self.text_output.bind(size=lambda w, s: setattr(w, 'text_size', (s[0] - 20, None)))
        root.add_widget(self.text_output)
        btn = Button(text="choose image to convert", background_color=[0.2, 0.2, 1, 1], size_hint=(0.5, 0.1))
        btn.bind(on_press=self.open_chooser)
        root.add_widget(btn)
        btn2 = Button(text="convert", background_color=[0.2, 0.2, 1, 1], size_hint=(0.5, 0.1))
        btn2.bind(on_press=self.convert)
        root.add_widget(btn2)
        return root

    def open_chooser(self, *_):
        content = BoxLayout(orientation='vertical')
        start = os.path.dirname(self.file_name) if self.file_name else DATASET_DIR
        chooser = FileChooserIconView(path=start, filters=['*.jpg', '*.jpeg', '*.png', '*.gif'])
        buttons = BoxLayout(size_hint_y=None, height=44)
        ok = Button(text="Select")
        cancel = Button(text="Cancel")
        buttons.add_widget(ok)
        buttons.add_widget(cancel)
        content.add_widget(chooser)
        content.add_widget(buttons)
        popup = Popup(title="Choose an image", content=content, size_hint=(0.9, 0.9))

        def select(*_):
            if chooser.selection:
                self.file_name = chooser.selection[0]
                self.image_source.source = self.file_name
                self.image_source.opacity = 1
                self.text_output.text = ''
            popup.dismiss()

        ok.bind(on_press=select)
        chooser.bind(on_submit=lambda *_: select())
        cancel.bind(on_press=popup.dismiss)
        popup.open()

    def convert(self, *_):
        if not self.file_name:
            self.text_output.text = 'Choose an image first.'
            return
        self.text_output.text = image_method(self.file_name)


if __name__ == "__main__":
    MainApp().run()
