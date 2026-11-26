import logging
import os
import threading
import time

# Kivy logs at debug level through Python's root logger, which would also print TensorFlow's
# internal debug messages: keep those quiet.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
for _name in ("tensorflow", "absl", "keras", "h5py"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from kivy.app import App
from kivy.clock import mainthread
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


class MainApp(App):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'convert handwritten text to typed text'
        self.file_name = ""
        self.engine = None                    # the app module, once TensorFlow and the model are loaded
        self.ready = threading.Event()
        self.load_error = None

    def build(self):
        root = GridLayout(cols=2)
        self.image_source = Image(source='', size_hint=(0.5, 0.9), fit_mode="contain", opacity=0)
        root.add_widget(self.image_source)
        self.text_output = Label(text='Loading the recognition model...', size_hint=(0.5, 0.9),
                                 halign='left', valign='middle')
        self.text_output.bind(size=lambda w, s: setattr(w, 'text_size', (s[0] - 20, None)))
        root.add_widget(self.text_output)
        btn = Button(text="choose image to convert", background_color=[0.2, 0.2, 1, 1], size_hint=(0.5, 0.1))
        btn.bind(on_press=self.open_chooser)
        root.add_widget(btn)
        self.convert_btn = Button(text="convert", background_color=[0.2, 0.2, 1, 1], size_hint=(0.5, 0.1))
        self.convert_btn.bind(on_press=self.convert)
        root.add_widget(self.convert_btn)
        return root

    def on_start(self):
        # Importing TensorFlow and loading the model takes a few seconds: do it in the background
        # so the window opens straight away.
        threading.Thread(target=self._load_engine, daemon=True).start()

    def _load_engine(self):
        try:
            import numpy as np
            import app
            model = app.get_model()
            model(np.zeros((1, 28, 28, 1), np.float32), training=False)   # warm up
            self.engine = app
        except Exception as e:  # shown in the window instead of crashing it
            self.load_error = e
        self.ready.set()
        self._loaded()

    @mainthread
    def _loaded(self):
        if self.load_error is not None:
            self.text_output.text = f'Could not load the model: {self.load_error}'
        elif self.text_output.text.startswith('Loading'):
            self.text_output.text = f'Model ready ({self.engine.model_name()} CNN). Choose an image.'

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
        self.convert_btn.disabled = True
        self.text_output.text = 'Converting...' if self.ready.is_set() else 'Loading the model, then converting...'
        threading.Thread(target=self._convert, args=(self.file_name,), daemon=True).start()

    def _convert(self, path):
        self.ready.wait()
        if self.engine is None:
            self._show(f'Could not load the model: {self.load_error}')
            return
        try:
            start = time.time()
            text = self.engine.image_method(path)
            self._show(text or '(no text found)', time.time() - start)
        except Exception as e:
            self._show(f'Could not convert this image: {e}')

    @mainthread
    def _show(self, text, seconds=None):
        self.text_output.text = text
        self.convert_btn.disabled = False
        if seconds is not None:
            self.title = f'convert handwritten text to typed text  ({seconds:.1f} s)'


if __name__ == "__main__":
    MainApp().run()
