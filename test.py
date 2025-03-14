
import random
class Store:
    
    def __init__(self):
        self.values = []
        self.map = {}
    def insert(self, value):
        if value in self.map:
            return
        self.values.append(value)
        self.map[value] = len(self.values) -1
    def remove(self, value):
        if value not in self.map:
            return
        last_value = self.values[-1]
        index= self.map[value]
        self.values[-1] =value
        self.values[index] = last_value
        self.values.pop()
        del(self.map[value])
    def random(self):
        print(random.choice(self.values))
                
t = Store()       
t.insert(3)
t.insert(5)
t.insert(5)
t.insert(1)
t.insert(1)
t.remove(3)
t.random()
print(t.map)
print(t.values)