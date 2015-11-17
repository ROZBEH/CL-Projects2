import sys
from collections import deque
from collections import defaultdict
import random
import copy

LEFT = "left"
RIGHT = "right"
SHIFT  = "shift"
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

    def hasJustRoot(self):
        if self._l[0].word == "ROOT":
            return True
        return False

class Queue:
    def __init__(self):
        self._l = list()

    def __iter__(self):
        return iter(self._l)

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
    def update(self, x, y, total):
        for feat,val in x.iteritems():
            if val != 0.:
                self[feat] += y * val
                total[feat] += self[feat]

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
    if not queue.isEmpty():
        feats[queue.getFirst().word] = 1
        feats[queue.getFirst().ID] = 1
        feats[queue.getFirst().pos] = 1
        feats[stack.getFirst().pos+"+"+queue.getFirst().word] = 1

    if stack.getsize() > 1:
        #feats[stack.getFirst().word] = 1
        feats[stack.getFirst().ID] = 1
        feats[stack.getFirst().pos] = 1
        feats[stack.getSecond().word+stack.getFirst().word] = 1
        feats[stack.getSecond().word+stack.getFirst().pos] = 1
        feats[stack.getSecond().pos+stack.getFirst().word] = 1
        feats[stack.getSecond().pos+stack.getFirst().pos] = 1
        

    if not queue.isEmpty() and stack.getsize() > 1:
        feats[stack.getFirst().word+queue.getFirst().word] = 1
        feats[stack.getFirst().pos+queue.getFirst().pos] = 1
        #feats[stack.getFirst().pos1+"+"+queue.getFirst().pos1] = 1

    #print feats
    return feats


def oracle(stack,queue):       
    if stack.hasOverTwo()  and stack.getFirst().ID == stack.getSecond().head and stack.getSecond().unproc == 0: 
        action(LEFT,queue,stack)
        return LEFT
    else: 
        if stack.hasOverTwo() and stack.getFirst().head == stack.getSecond().ID and stack.getFirst().unproc == 0:
            action(RIGHT,queue,stack)
            return RIGHT
        else: 
            x = action(SHIFT,queue,stack)
            if x == -1:
                return x
            return SHIFT


def make_correct_action(stack, queue):
    if stack.getSecond() is None:
        return SHIFT
    
    elif stack.getSecond().head == stack.getFirst().ID and stack.getSecond().unproc == 0:
        return LEFT
    
    elif stack.getFirst().head == stack.getSecond().ID and stack.getFirst().unproc == 0:
            return RIGHT
    else:
        return SHIFT
    

def action(correct, queue, stack):
    #@print queue.getsize()   
    if correct == SHIFT:
            if queue.getsize() == 0:
                return -1
            stack.push(queue.deQueue())
    elif correct == RIGHT:
            stack.getSecond().unproc -= 1
            stack.pop()
    else:
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

def train_parser(sentence,w_s,w_l,w_r,w_s_avg,w_l_avg,w_r_avg):
    stack = Stack([Token(str(0), "ROOT", "ROOT", "ROOT",-1)])
    #print stack.getsize()
    queue = Queue()

    tmp_unproc = Weights()
    for word in sentence:
        queue.enQueue(Token(word[0],word[1],word[3],word[4],word[6]))
        tmp_unproc[(word[6])] += 1

    for each in queue:
        each.unproc = tmp_unproc[each.ID]
    
    
    #print stack1.getsize()
    #print queue1.getsize()
   

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
                
       
        corr = oracle(stack,queue)
        if corr == -1:
            raise ValueError('projective sentence')
        if corr != ans:
            if ans == LEFT:
                w_l.update(feats,-1,w_l_avg)
            if ans == RIGHT:
                w_r.update(feats,-1,w_r_avg)
            if ans == SHIFT:
                w_s.update(feats,-1,w_s_avg)

            if corr == LEFT:
                w_l.update(feats,1,w_l_avg)
            if corr == RIGHT:
                w_r.update(feats,1,w_r_avg)
            if corr == SHIFT:
                w_s.update(feats,1,w_s_avg)
        #print w_s
        #print str(corr)+" from train_parser"
        #action(corr,queue,stack)
        #print cnt
        #print corr
        #print queue.getsize()

        #print "success"



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
    w_l_avg = Weights()
    w_r_avg = Weights()
    w_s_avg = Weights()
    sentences = read_conll(train)
    print len(sentences)
    N = 7
    cnt = 0
    for i in range(0,N):
        #random.shuffle(sentences)
        for sentence in sentences:
            try:
                train_parser(sentence,w_s,w_l,w_r,w_s_avg,w_l_avg,w_r_avg)
                cnt += 1
            except ValueError:
                #print "here"
                continue
            '''
            for feat,val in w_s.iteritems():
                w_s_avg[feat] += val
            for feat,val in w_r.iteritems():
                w_r_avg[feat] += val
            for feat,val in w_l.iteritems():
                w_l_avg[feat] += val
            '''
        print i
        
            #print "==========================="
    #train(sentences)
    T = len(sentences)

    
    for feat,val in w_s_avg.iteritems():
        w_s_avg[feat] = w_s_avg[feat]*1.0/N*cnt

    for feat,val in w_l_avg.iteritems():
        w_l_avg[feat] = w_l_avg[feat]*1.0/N*cnt

    for feat,val in w_r_avg.iteritems():
        w_r_avg[feat] = w_r_avg[feat]*1.0/N*cnt
    
    
    #print w_s_avg
    print "training done"
    test_sentences = read_conll(test)
    f_out = open(out,"w")
    for sentence in test_sentences:
        #print sentence
        #evaluate(sentence,f_out,w_s_avg,w_l_avg,w_r_avg)
        evaluate(sentence,f_out,w_s,w_l,w_r)

    
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
