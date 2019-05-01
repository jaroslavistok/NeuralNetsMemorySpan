import datetime


class Logger:
    def __init__(self, file_name):
        self.file_name = file_name.datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def log_receptive_field(self, receptive_field):
        with open(self.file_name, 'a') as file:
            file.write('Receptive field: {}'.format(receptive_field))

    def log_value(self, key, value):
        with open(self.file_name, 'a') as file:
            file.write('{}: {}'.format(key, value))

