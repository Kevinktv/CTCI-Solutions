from __future__ import annotations
from typing import Optional, Any

class Tree:
    root: Optional[Any]
    children: list[Tree]

    def __init__(self, item: Any):
        self.root = item
        self.children = []


# Binary tree traversals and definiition etc
class BinaryTree:
    root: Optional[Any]
    left: Optional[BinaryTree]
    right: Optional[BinaryTree]

    def __init__(self, item: Any):
        self.root = item
        self.left = None
        self.right = None

def inOrder(node: BinaryTree):
    if node is None:
        return
    else:
        inOrder(node.left)
        print(node.root)
        inOrder(node.right)

# def inOrderIterative(node: BinaryTree):
#     stack = []
#     stack.append(node)
#     while stack != []:
#         latest = stack[-1]
#         if latest.left:
#             stack.append(latest.left)
#         else: # leftmost node
#             print()


def preOrder(node: BinaryTree):
    if node is None:
        return
    else:
        print(node.root)
        preOrder(node.left)
        preOrder(node.right)

def postOrder(node: BinaryTree):
    if node is None:
        return
    else:
        print(node.left)
        print(node.right)
        print(node)

binaryTree = BinaryTree(1)
binaryTree.left = BinaryTree(2)
binaryTree.right = BinaryTree(3)
binaryTree.left.left = BinaryTree(4)
binaryTree.left.right = BinaryTree(5)
binaryTree.right.left = BinaryTree(6)
binaryTree.right.right = BinaryTree(7)


# Heap implementation using Array starting at index 1

class MinHeap:
    items: list[int]
    size: int

    def __init__(self):
        self.items = [None]
        self.size = 0

    def insert(self, item: int):
        self.size += 1
        self.items.append(item)

        index = self.size
        parentIndex = index // 2

        while index != 1 and self.items[index] < self.items[parentIndex]:
            self.items[index], self.items[parentIndex] = self.items[parentIndex], self.items[index]
            index = index // 2
            parentIndex = index // 2

    def peek(self):
        return self.items[1]

    def extractMin(self):
        if self.size == 0:
            return
        item = self.items[1]
        self.items[1] = self.items[self.size]
        self.items.pop()
        self.size -= 1
        if self.size == 0:
            return item

        self.bubbleDown(1)
        return item

    def bubbleDown(self, index: int):
        """
        Assuming left and right children of the particular index node follows heap prop,
        bubbleDown satisfies heap prop of node rooted at index.
        :param index:
        :return:
        """
        left = 2 * index
        right = 2 * index + 1
        minim = index

        if left <= self.size and self.items[left] < self.items[index]:
            minim = left

        if right <= self.size and self.items[right] < self.items[minim]:
            minim = right

        if minim != index:
            self.items[minim], self.items[index] = self.items[index], self.items[minim]
            self.bubbleDown(minim)
