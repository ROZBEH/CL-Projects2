import sys

class DefaultList(list):
    """A list that returns a default value if index out of bounds."""
    def __init__(self, default=None):
        self.default = default
        list.__init__(self)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return self.default

def pad_tokens(tokens):
    tokens.insert(0, '<start>')
    tokens.append('ROOT')


def parse(words):
    n = len(words)
    stack = [1]
 
'''
calculate featres as map taking stack and queue as input. features are key value pairs where key is feature name and value is feature defined
on her webpage
'''

def get_features(stack,queue):

    feats = {}
    #calculate features

    return feats

def read_conll(loc):
    f = open(loc,"r")
    lines = f.readlines()
    ret = []
    for line in lines:
        line = line.strip()
        attrs = line.split("\t")
        print attrs
        ret.append(attrs)

    return ret

def main(train,test,out):
    sentences = list(read_conll(train))
    print sentences

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
