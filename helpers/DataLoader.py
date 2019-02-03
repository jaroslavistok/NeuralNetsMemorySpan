class DataLoader:
    @staticmethod
    def load_data(filename):
        file_path = '../data/sequences/' + filename
        with open(file_path) as data_file:
            loaded_string = data_file.read()
        return loaded_string
