import pickle

def unpickle_file():
    data = []
    with open ("output.pkl", "rb") as pickle_off:
        while True:
            try:
                print(pickle.load(pickle_off, encoding="latin1"))
            except EOFError:
                break


def csv_to_pkl():
    return None

def pkl_to_csv():
    input_file = open("meld_sentences.pkl", "rb")
    new_file = pickle.load(input_file)
    input_file.close()
    print(new_file)

if __name__ == "__main__":
    unpickle_file()