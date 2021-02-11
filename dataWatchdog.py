import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class DataWatchdog():

    def __init__(self, path, trainManager, patterns="*.wav"):
        print('starting data watchdog')
        self.trainManager = trainManager

        ignore_patterns = ""
        ignore_directories = True
        case_sensitive = True
        self.event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

        self.event_handler.on_created = self.on_created
        self.event_handler.on_deleted = self.on_deleted
        self.event_handler.on_modified = self.on_modified
    
        self.observer = Observer()
        self.observer.schedule(self.event_handler, path, recursive=False)
        self.observer.start()


    def on_created(self, event):
        print(f"{event.src_path} has been created!")

    def on_deleted(self, event):
        print(f"{event.src_path} deleted")

    def on_modified(self, event):
        print(f"{event.src_path} has been modified, loading training example")
        self.trainManager.load_train_data(event.src_path)

    def stop_observer(self):
        self.observer.stop()
        self.observer.join()