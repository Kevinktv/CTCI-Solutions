"""
Stacks and queues
"""

from __future__ import annotations
from typing import Optional


class Stack:
    items: list

    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.items == []:
            return
        else:
            return self.items.pop()

    def peek(self):
        if self.items != []:
            return self.items[-1]
        return None

    def __len__(self):
        return len(self.items)


# 3.2 Stack min: How would you design a stack which, in addition to push and pop, has a function min
# which returns the minimum element? Push, pop and min should all operate in 0(1) time.

class minStack(Stack):
    minvals: list  # Contains the list of minimum items till that point. When a new item

    # smaller is added, it is also added to minvals

    def __init__(self):
        super().__init__()
        self.minvals = []

    def push(self, item: int):
        super().push(item)
        if self.minvals == [] or item < self.minvals[-1]:
            self.minvals.append(item)
        else:
            self.minvals.append(self.minvals[-1])

    def pop(self):
        popped = super().pop()
        if self.minvals != []:
            self.minvals.pop()
        return popped

    def minimum(self):
        return self.minvals[-1]


# 3.3
# Stack of Plates: Imagine a (literal) stack of plates. If the stack gets too high, it might topple.
# Therefore, in real life, we would likely start a new stack when the previous stack exceeds some
# threshold. Implement a data structure SetOfStacks that mimics this. SetO-fStacks should be
# composed of several stacks and should create a new stack once the previous one exceeds capacity.
# SetOfStacks. push() and SetOfStacks. pop() should behave identically to a single stack
# (that is, pop () should return the same values as it would if there were just a single stack).
# FOLLOW UP
# Implement a function popAt ( int index) which performs a pop operation on a specific sub-stack.

class StackOfPlates:
    stacks: list[Stack]
    threshold: int

    def __init__(self, threshold: int):
        self.stacks = [Stack()]
        self.threshold = threshold

    def push(self, item: int):
        stck = self.stacks[-1]

        if len(stck) == self.threshold:
            self.stacks.append(Stack())
            stcknew = self.stacks[-1]
            stcknew.push(item)
        else:
            stck.push(item)

    def pop(self):
        lenOfStacks = len(self.stacks)
        stck = self.stacks[-1]

        if lenOfStacks == 1:  # Only 1 stack
            if len(stck) == 0:  # Empty first stack
                return
            else:
                return stck.pop()
        else:  # More than 1 stack
            if len(stck) == 1:
                item = stck.pop()
                self.stacks.pop()
                return item
            else:
                return stck.pop()

    def popAtIndex(self, index: int):
        """
        This one, u just delete item so not all stacks at full capa
        :param index:
        :return:
        """
        if index > len(self.stacks) - 1:
            raise IndexError

        stck = self.stacks[index]
        if len(stck) == 1:
            item = stck.pop()
            self.stacks.pop(index)
            return item
        else:
            return stck.pop()


# 3.4:
# Queue via Stacks: Implement a MyQueue class which implements a queue using two stacks.

class QueueViaStacks:
    stck: Stack
    stckRev: Stack

    def __init__(self):
        self.stck = Stack()
        self.stckRev = Stack()

    def enqueue(self, item):
        self.stck.push(item)

    def shiftIntoRev(self):
        if len(self.stckRev) == 0:
            while len(self.stck) != 0:
                self.stckRev.push(self.stck.pop())

    def dequeue(self):
        self.shiftIntoRev()
        return self.stckRev.pop()

    def peek(self):
        self.shiftIntoRev()
        return self.stckRev.peek()


# 3.5
# Sort Stack: Write a program to sort a stack such that the smallest items are on the top. You can
# use
# an additional temporary stack, but you may not copy the elements into any other data structure
# (such as an array). The stack supports the following operations: push, pop, peek, and is Empty.

def sortStack(stack: Stack):
    """
    One implem I can think of is selection sort. I take the largest elem in stack and
    insert it into a temp stack. O(n^2). But needs 3 stacks.

    But u can do the sorting in two stacks by inserting each item popped from s1 to s2
    in the correct place. Basically insertion sort.
    :param stack:
    :return:
    """
    temp = Stack()

    while len(stack) != 0:
        item = stack.pop()
        if temp.peek() == None:
            temp.push(item)
        else:
            ctr = 0
            while temp.peek() is not None and item > temp.peek():
                stack.push(temp.pop())
                ctr += 1
            temp.push(item)
            while ctr != 0:
                temp.push(stack.pop())
                ctr -= 1

    return temp


#  3.6

class Animal:
    name: str
    order: int

    def __init__(self, name: str):
        self.name = name
        self.order = -1

    def setOrder(self, order: int):
        self.order = order

    def getOrder(self):
        return self.order


class Dog(Animal):
    nextDog: Optional[Dog]

    def __init__(self, name: str):
        super().__init__(name)
        self.nextDog = None


class Cat(Animal):
    nextCat: Optional[Cat]

    def __init__(self, name: str):
        super().__init__(name)
        self.nextCat = None


class AnimalShelter:
    """
    An animal shelter, which holds only dogs and cats, operates on a strictly"first in, first
    out" basis. People must adopt either the "oldest" (based on arrival time) of all animals at the shelter,
    or they can select whether they would prefer a dog or a cat (and will receive the oldest animal of
    that type). They cannot select which specific animal they would like. Create the data structures to
    maintain this system and implement operations such as enqueue, dequeueAny, dequeueDog,
    and dequeueCat. You may use the built-in Linked list data structure.
    """
    # BTW my code only works till order = 99999 cuz that is the max I set
    headDog: Optional[Dog]
    tailDog: Optional[Dog]
    headCat: Optional[Cat]
    tailCat: Optional[Cat]
    globalOrder: int

    def __init__(self):
        self.headDog = None
        self.headCat = None
        self.tailDog = None
        self.tailCat = None
        self.globalOrder = 0

    def enqueue(self, a: Animal):
        self.globalOrder += 1
        a.setOrder(self.globalOrder)

        if isinstance(a, Dog):
            if self.headDog is None:
                self.headDog = a
                self.tailDog = a
            else:
                self.tailDog.nextDog = a
                self.tailDog = a

        elif isinstance(a, Cat):
            if self.headCat is None:
                self.headCat = a
                self.tailCat = a
            else:
                self.tailCat.nextCat = a
                self.tailCat = a

        else:
            return

    def dequeueCat(self):
        if self.headCat is None:
            return

        cat = self.headCat
        if self.tailCat is cat: # Then only one node in it so empty queue
            self.headCat, self.tailCat = None, None
            return cat
        else:
            self.headCat = cat.nextCat
            return cat

    def dequeueDog(self):
        if self.headDog is None:
            return

        dog = self.headDog
        if self.tailDog is dog:
            self.headDog, self.tailDog = None, None
            return dog
        else:
            self.headDog = dog.nextDog
            return dog

    def dequeueAny(self):
        if self.headDog is None and self.headCat is None:
            return

        dogOrder = self.headDog.order if self.headDog else 99999
        catOrder = self.headCat.order if self.headCat else 99999
        if dogOrder > catOrder:
            return self.dequeueCat()
        else:
            return self.dequeueDog()
