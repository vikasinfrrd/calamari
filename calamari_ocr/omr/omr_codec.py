from calamari_ocr.ocr import Codec


class SimpleAncientCodec(Codec):
    def __init__(self, charset=[]):
        charset = [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",   # Note positions
            "(", ")", " ",                                      # Compound/connected notes
            "F1", "F3", "F5", "F7",                             # F-Clef with position
            "C1", "C3", "C5", "C7",                             # C-Clef with position
            ]
        super().__init__(charset)

    def encode(self, s):
        index = 0
        out = []
        while index < len(s):
            found = False
            for key, value in self.char2code.items():
                if len(key) == 0:
                    continue  # blank
                if s[index:index+len(key)] == key:
                    out.append(value)
                    index += len(key)
                    found = True
                    break

            if found:
                continue

            else:
                raise Exception("Could not parse: '{}'".format(s[index:]))

        return out
