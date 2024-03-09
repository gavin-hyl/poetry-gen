import re

def parse_poems(filename):
    '''
    Parses shakespeare.txt and spenser.txt into a list of strings, each string is a poem.
    '''
    with open(filename, 'r') as file:
        text = file.read()
    shakespeare_pattern = r'[\d]{1,}\n+(\D+)\n{2}'
    spenser_pattern = r'\n\n\s+(\D+?)\n\n'
    
    poem_matches = re.findall(spenser_pattern if filename == 'spenser.txt' else shakespeare_pattern, text) 
    poems = []
    for match in poem_matches:
        poems.append(match)
    print(f'Number of poems found: {len(poems)}')
    return poems


def separate_words(poems: list[str]):
    '''
    Takes in a list of poems (strings) and returns a 3D list of the words, with
    the indices being [poem_number][line_number][word_number] (a newline is added)
    '''
    # hyphens and apostrophes treated as joining two parts of a word
    # and punctuation ,.?!:; are treated as separated tokens
    pattern = r'([\w\'-]+\b|[,.?!:;])'
    all_words = []
    for poem in poems:
        lines = poem.splitlines()
        words = []
        for idx, line in enumerate(lines):
            words.append(re.findall(pattern, line) + ['\n'])
        all_words.append(words)
    return all_words

def word_idx_mappings(words: list[list[list[str]]]):
    '''
    Returns the word to index mapping and the index to word mapping (including END)
    '''
    word_to_idx = {}
    idx = 0
    for poem in words:
        for line in poem:
            for word in line:
                if word_to_idx.get(word) is None:
                    word_to_idx.update({word: idx})
                    idx += 1
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    print(f'Number of unique observations (including punctuation and newline): {len(word_to_idx)}')
    return word_to_idx, idx_to_word

def numerize_words(words, word_to_idx: dict):
    word_labels = []
    for poem in words:
        word_labels.append([])
        for line in poem:
            word_labels[-1].append([])
            for word in line:
                word_labels[-1][-1].append(word_to_idx.get(word))
    return word_labels

def parse_syllable_dict(filename='syllable_dict.txt'):
    syllable_dict = {}
    with open(filename, 'r') as file:
        text = file.read().splitlines()
    for line in text:
        tokens = line.split(' ')
        syllable_dict.update({tokens[0] : [int(t[1] if len(t) > 1 else t) for t in tokens[1:]]})
    return syllable_dict
print(parse_syllable_dict())
# parse_syllable_dict()