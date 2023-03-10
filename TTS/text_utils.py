from symbols import symbols


class TextCleaner:
    def __init__(self):
        dicts = {}
        for i in range(len((symbols))):
            dicts[symbols[i]] = i
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes