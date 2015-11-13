import sys
from collections import deque
import random

LEFT = 1
RIGHT = 2
SHIFT  = 3
UNK = 4


class Stack:
    def __init__(self, l=list()):
        self._l = l
    def getsize(self):
        return len(self._l)

    def push(self, item):
        self._l.append(item)

    def pop(self):
        return self._l.pop(-1)

    def pop_second(self):
        return self._l.pop(-2)

    def getFirst(self):
        try:
            return self._l[-1]
        except IndexError:
            return None

    def getSecond(self):
        try:
            return self._l[-2]
        except IndexError:
            return None

    def hasOverTwo(self):
        return len(self._l) >= 2


class Queue:
    def __init__(self):
        self._l = list()

    def getsize(self):
        return len(self._l)

    def enQueue(self, item):
        self._l.append(item)

    def deQueue(self):
        return self._l.pop(0)

    def isEmpty(self):
        return len(self._l) == 0

    def getIter(self):
        for item in self._l:
            yield item

    def getFirst(self):
        return self._l[0]

class Weights(dict):
    # default all unknown feature values to zero
    def __getitem__(self, idx):
        if self.has_key(idx):
            return dict.__getitem__(self, idx)
        else:
            return 0.

    # given a feature vector, compute a dot product
    def dotProduct(self, x):
        dot = 0.
        for feat,val in x.iteritems():
            dot += val * self[feat]
        return dot

    # given an example _and_ a true label (y is +1 or -1), update the
    # weights according to the perceptron update rule (we assume
    # you've already checked that the classification is incorrect
    def update(self, x, y):
        for feat,val in x.iteritems():
            if val != 0.:
                self[feat] += y * val

class Token:
    def __init__(self, ID, word, pos, pos1, head, unproc=0):
        self.ID = ID
        self.word = word
        self.pos = pos
        self.pos1 = pos1
        self.head = head
        self.unproc = unproc



 
'''
calculate featres as map taking stack and queue as input. features are key value pairs where key is feature name and value is feature defined
on her webpage
'''

def get_features(stack,queue):

    feats = Weights()
    #calculate features
    if queue.isEmpty():
        qword = ''
        qid = ''
        qpos = ''
        
    else:
        qword = queue.getFirst().word
        qid = queue.getFirst().ID
        qpos= queue.getFirst().pos

    if stack.getFirst() != None:
        sid = stack.getFirst().ID
        sword = stack.getFirst().word
        spos = stack.getFirst().pos

    else:
        sid = ''
        sword = ''
        spos = ''

    feats[sid] = 1
    feats[qid] = 1
    feats[spos] = 1
    feats[qpos] = 1
    feats[sword+"+"+qword] = 1
    feats[spos +"+"+qpos] = 1
    return feats



def make_correct_action(stack, queue):
    if stack.getSecond() is None:
        return SHIFT
    
    elif stack.getSecond().head == stack.getFirst().ID:
        return LEFT
    
    elif stack.getFirst().head == stack.getSecond().ID:
        return RIGHT
    else:
        if queue.getsize() != 0:
            return SHIFT

def action(correct, queue, stack):
    if correct == SHIFT:
        stack.push(queue.deQueue())
    elif correct == RIGHT:
            if stack.getSecond() != None:
                stack.getSecond().unproc -= 1
            stack.pop()
    else:
        if stack.getFirst() != None:
            stack.getFirst().unproc -= 1
        stack.pop_second()


def action_test(predicted, queue, stack):

    if predicted == SHIFT:
        stack.push(queue.deQueue())
    elif predicted == RIGHT:
       
            stack.getFirst().head = stack.getSecond().ID
            stack.pop()
    else:
            stack.getSecond().head = stack.getFirst().ID
            stack.pop_second()

#id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel

def train_parser(sentence,w_s,w_l,w_r):
    stack = Stack([Token(0, "ROOT", "ROOT", "ROOT",-1)])
    queue = Queue()

    for word in sentence:
        queue.enQueue(Token(word[0],word[1],word[3],word[4],word[6]))
    
    
    while not queue.isEmpty() or stack.hasOverTwo():
        #TODO this
        feats = get_features(stack,queue)

        s_s = w_s.dotProduct(feats)
        s_l = w_l.dotProduct(feats)
        s_r = w_r.dotProduct(feats)

        if s_s >= s_l and s_s >= s_r and not queue.isEmpty():
            ans = SHIFT
        else:
            if s_l >= s_r:
                ans = LEFT
               
            else:
                ans = RIGHT
                
        
        corr = make_correct_action(stack, queue)
        if corr != ans:
            if ans == LEFT:
                w_l.update(feats,-1)
            if ans == RIGHT:
                w_r.update(feats,-1)
            if ans == SHIFT:
                w_s.update(feats,-1)

            if corr == LEFT:
                w_l.update(feats,1)
            if corr == RIGHT:
                w_r.update(feats,1)
            if corr == SHIFT:
                w_s.update(feats,1)
        #print w_s
        action(corr,queue,stack)
        
        #print corr
        #print queue.getsize()

        



def read_conll(loc):
    sentences = []
    for sent_str in open(loc).read().strip().split('\n\n'):
        lines = [line.split() for line in sent_str.split('\n')]
        sentences.append(lines)

    return sentences

        

def evaluate(sentence,f,w_s,w_l,w_r):

    stack = Stack([Token(0, "ROOT", "ROOT", "ROOT",-1)])
    queue = Queue()

    tokmap = {}
    for word in sentence:
        t = Token(word[0],word[1],word[3],word[4],word[6])
        queue.enQueue(t)
        tokmap[word[0]] = t
    
    
    while not queue.isEmpty() or stack.hasOverTwo():
        #TODO this
        feats = get_features(stack,queue)
        s_s = w_s.dotProduct(feats)
        s_l = w_l.dotProduct(feats)
        s_r = w_r.dotProduct(feats)

        if not queue.isEmpty() and s_r <= s_s and s_l <= s_s:
            ans = SHIFT
    
        elif s_r <= s_l and stack.hasOverTwo():
            ans = LEFT
    
        elif stack.hasOverTwo():
            ans = RIGHT
        elif not queue.isEmpty():
            ans = SHIFT
        else:
            ans = UNK

        if ans == UNK:
            break
        
        action_test(ans,queue,stack)

    for word in sentence:
        for i in range(0,len(word)):
            if i == 6:
                f.write(str(tokmap[word[0]].head))
            else:
                f.write(word[i])
            f.write("\t")
        f.write("\n")

    f.write("\n")

def main(train,test,out):
    w_l = Weights()
    w_r = Weights()
    w_s = Weights()
    sentences = read_conll(train)
    for i in range(0,100):
        random.shuffle(sentences)
        for sentence in sentences:
            train_parser(sentence,w_s,w_l,w_r)
            #print "==========================="
    #train(sentences)
        
    test_sentences = read_conll(test)
    f_out = open(out,"w")
    for sentence in test_sentences:
        #print sentence
        evaluate(sentence,f_out,w_s,w_l,w_r)

    
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
