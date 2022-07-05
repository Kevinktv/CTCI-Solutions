from __future__ import annotations
from typing import Optional, Any
from collections import deque


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


# Trie
class Trie:
    children: dict[str, Trie]
    endOfWord: bool

    def __init__(self):
        self.children = {}
        self.endOfWord = False

def insertInTrie(word: str, head: Trie):
    curr = head
    for i in word:
        if i not in curr.children:
            curr.children[i] = Trie()
            curr = curr.children[i]
        else:
            curr = curr.children[i]
    curr.endOfWord = True

def searchInTrie(word: str, head: Trie):
    curr = head
    for i in word:
        if i not in curr.children:
            return False
        else:
            curr = curr.children[i]

    return curr.endOfWord

def deleteInTrie(word: str, head: Trie):
    if not searchInTrie(word, head):
        return

    deleteInTrieHelper(word, head, 0)

def deleteInTrieHelper(word: str, head: Trie, index: int):
    if len(word) == index:
        if head.endOfWord:
            head.endOfWord = False
        return len(head.children) == 0

    node = head.children[word[index]]
    shouldDel = deleteInTrieHelper(word, node, index + 1)
    if shouldDel:
        del head.children[word[index]]
        return len(head.children) == 0

    return False

class Vertex:
    item: Optional[Any]
    neighbours: list[Vertex]

    def __init__(self, item: Any):
        self.item = item
        self.neighbours = []

    def addNeighbour(self, v: Vertex):
        if v not in self.neighbours:
            self.neighbours.append(v)


class GraphUndirected:
    vertices: dict[str, Vertex]

    def __init__(self):
        self.vertices = {}

    def addVertex(self, item):
        if item not in self.vertices:
            self.vertices[item] = Vertex(item)

    def addEdge(self, item1: Any, item2: Any):
        v1 = self.vertices[item1]
        v2 = self.vertices[item2]
        v1.addNeighbour(v2)
        v2.addNeighbour(v1)

    def dfs(self):
        color = {}
        for v in self.vertices:
            color[v] = "white"

        for v in self.vertices:
            if color[v] == "white":
                self.dfsHelper(v, color)

    def dfsHelper(self, v: str, color: dict):
        """
        Does dfs on node v
        :param v:
        :param color:
        :return:
        """
        print(v) # Pre procesing
        vertex = self.vertices[v]
        color[v] = "grey"
        for adj in vertex.neighbours:
            if color[adj.item] == "white":
                self.dfsHelper(adj.item, color)

        color[v] = "black"

    def bfs(self, start: str):
        color = {}
        dist = {}
        for v in self.vertices:
            color[v] = "white"
            dist[v] = None

        q = deque()
        q.append(start)
        dist[start] = 0

        while len(q) != 0:
            item = q.popleft()
            vert = self.vertices[item]
            print(item) # Processing current node
            for v in vert.neighbours:
                if color[v.item] == "white":
                    color[v.item] = "grey"
                    dist[v.item] = dist[item] + 1
                    q.append(v.item)

            color[item] = "black"

        return dist


class GraphDirected:
    vertices: dict[str, Vertex]
    def __init__(self):
        self.vertices = {}

    def addVertex(self, item):
        if item not in self.vertices:
            self.vertices[item] = Vertex(item)

    def addEdge(self, item1, item2):
        v1 = self.vertices[item1]
        v2 = self.vertices[item2]
        if v2 not in v1.neighbours:
            v1.addNeighbour(v2)


# 4.1
def routeBetweenNodes(g: GraphDirected, item1: str, item2: str):
    """
    Route Between Nodes: Given a directed graph, design an algorithm to find out whether there is a
    route between two nodes.
    :param g:
    :return:
    """
    if item1 not in g.vertices or item2 not in g.vertices:
        return False

    return routeBetweenHelper(g, item1, item2, set())

def routeBetweenHelper(g: GraphDirected, item1: str, item2: str, visited: set):
    v1 = g.vertices[item1]
    v2 = g.vertices[item2]

    if v1 is v2:
        return True

    visited.add(v1)
    for v in v1.neighbours:
        if v not in visited:
            result = routeBetweenHelper(g, v.item, item2, visited)
            if result:
                return True
    return False


# 4.2
def minimalTree(arr: list):
    """
    Minimal Tree: Given a sorted (increasing order) array with unique integer elements, write an algorithm
    to create a binary search tree with minimal height.
    :param arr:
    :return:

    For creating a minimal tree, I needa have the root be the middle elem each time
    of the array.
    """

    start = 0
    end = len(arr) - 1
    return minimalTreeHelper(arr, start, end)

def minimalTreeHelper(arr: list, start: int, end: int):
    if end < start:
        return None
    else:
        mid = (start + end) // 2
        node = BinaryTree(arr[mid])
        node.left = minimalTreeHelper(arr, start, mid - 1)
        node.right = minimalTreeHelper(arr, mid + 1, end)

        return node


# 4.3
class Node:
    root: Any
    next: Optional[Node]

    def __init__(self, item: Any):
        self.root = item
        self.next = None


def makeLL(items: list[int]):
    flag = 0
    temp = None
    head = None
    for i in items:
        node = Node(i)
        if flag == 0:  # First item in list
            flag = 1
            temp = node
            head = node
        else:
            temp.next = node
            temp = node

    return head


def printLL(head: Node):
    while head is not None:
        print(head.root, " ")
        head = head.next

def listOfDepths(tree: BinaryTree):
    """
    List of Depths: Given a binary tree, design an algorithm which creates a linked list of all the nodes
    at each depth (e.g., if you have a tree with depth D, you'll have D linked lists).
    :param tree:
    :return:

    I am thinking of doing a bfs and then creating linked list of each level
    """
    q = deque()
    linkedLists = {}

    dist = {}
    dist[tree] = 0
    q.append(tree)
    while len(q) > 0:
        node = q.popleft()
        left = node.left
        right = node.right
        dist[left] = dist[node] + 1
        dist[right] = dist[node] + 1
        if left is not None:
            q.append(left)
        if right is not None:
            q.append(right)

        # Building the ll
        if dist[node] not in linkedLists:
            linkedLists[dist[node]] = Node(node.root)
        else:
            linkedNode = Node(node.root)
            linkedNode.next = linkedLists[dist[node]]
            linkedLists[dist[node]] = linkedNode

    return linkedLists


# 4.4
def checkBalanced(tree: BinaryTree):
    """
    Check Balanced: Implement a function to check if a binary tree is balanced. For the purposes of
    this question, a balanced tree is defined to be a tree such that the heights of the two subtrees of any
    node never differ by more than one.
    :param tree:
    :return:

    Well one implem is to go through every node and check the height difference but
    this seems to be a brute force approach and waay to slow.

    The implem below, I return at every point the height of the subtree and whether
    the subtree is balanced or not.
    """
    # if tree.left is None and tree.right is None:
    #     return [0, True]

    leftHeight, rightHeight, balLeft, balRight = -1, -1, True, True

    if tree.left is not None:
        leftHeight, balLeft = checkBalanced(tree.left)

    if tree.right is not None:
        rightHeight, balRight = checkBalanced(tree.right)


    largerHeight = leftHeight if leftHeight > rightHeight else rightHeight
    return [largerHeight + 1, balLeft and balRight and (abs(leftHeight - rightHeight) <= 1)]

#4.5
def validateBST(tree: BinaryTree):
    """
    Validate BST: Implement a function to check if a binary tree is a binary search tree.

    lefttree <= root <= right tree

    Well one thing I straightup think of is doing an in order traversal of the tree.
    For a binary search tree, it should come up sorted. This would be O(n).


    :param tree:
    :return:
    """
    if tree is None:
        return True

    if tree.left is None and tree.right is None:
        return True

    left = -99999 if tree.left is None else tree.left.root
    right = 99999 if tree.right is None else tree.right.root
    isRootSati = left <= tree.root < right

    return validateBST(tree.left) and validateBST(tree.right) and isRootSati


def validateBST2(tree: BinaryTree):
    """
    2nd implem using range of possible value for every node.
    :param tree:
    :return:
    """
    maxi = 99999
    mini = -99999
    return validateBST2Helper(tree, mini, maxi)

def validateBST2Helper(tree: BinaryTree, mini: int, maxi: int):
    if tree is None:
        return True

    b3 = mini <= tree.root < maxi
    if not b3:
        return False
    b1 = validateBST2Helper(tree.left, mini, tree.root)
    b2 = validateBST2Helper(tree.right, tree.root, maxi)

    return b1 and b2


class BinaryTreePar:
    parent: Optional[BinaryTreePar]
    left: Optional[BinaryTreePar]
    right: Optional[BinaryTreePar]
    root: Optional[Any]

    def __init__(self, item):
        self.parent = None
        self.left = None
        self.right = None
        self.root = item



# 4.6
def leftmost(tree: BinaryTreePar):
    temp = tree
    while temp.left is not None:
        temp = temp.left
    return temp

def successor(tree: BinaryTreePar):
    """
    Successor: Write an algorithm to find the "next" node (i.e., in-order successor) of a given node in a
    binary search tree. You may assume that each node has a link to its parent.
    :param tree:
    :return:

    Well one method is to do inorder and get the next elem but not efficient as O(n)

    Root (Has left and right child) - Leftmost node in right child
    Has only left child - Parent if node is in left of parent. Go up parents
    until u get to a parent from the left side or u reach the root without doing dat
    (In which case it is the rightmost node)
    (Dont care) Has only right child - Leftmost in right child
    """
    if tree.right is not None:
        return leftmost(tree.right)

    temp = tree
    par = temp.parent

    while par is not None and par.left is not temp:
        temp = par
        par = temp.parent

    return par

# 4.7
def hasCycle(g: GraphDirected):
    """
    Checks whether g has a cycle in it
    :param g:
    :return:
    """
    color = {}
    finish = []
    for u in g.vertices:
        color[u] = "white"

    stck = set()

    for u in g.vertices:
        if color[u] == "white":
            res = hasCycleHelper(g, u, color, finish, stck)
            if res == True:
                return []

    return finish

def hasCycleHelper(g: GraphDirected, u:Any, color: dict, finish: list, stck: set):
    stck.add(u)
    color[u] = "grey"

    for v in g.vertices[u].neighbours:
        if color[v.item] == "white":
            if hasCycleHelper(g, v.item, color, finish, stck):
                return True
        else:
            if v.item in stck:
                return True

    color[u] = "black"
    stck.remove(u)
    finish.append(u)
    return False



def buildOrder(projects: list, dependencies: list[list]):
    """
    Build Order: You are given a list of projects and a list of dependencies (which is a list of pairs of
    projects, where the second project is dependent on the first project). All of a project's dependencies
    must be built before the project is. Find a build order that will allow the projects to be built. If there
    is no valid build order, return an error.
    :param projects:
    :param dependencies:
    :return:

    This looks like a problem of topological sorting. So firstly it cannot contain
    cycles. If so no sort. Order it in decreasing order of finish time and no cycles.
    """
    g = GraphDirected()

    for v in projects:
        g.addVertex(v)

    for edge in dependencies:
        g.addEdge(edge[0], edge[1])

    res = hasCycle(g)
    res.reverse()
    return res

# 4.8
def isDescendant(tree: BinaryTree, node: BinaryTree):
    """
    Checks whether node is a descendant of tree
    :param tree:
    :param node:
    :return:
    """
    if tree is None:
        return False

    if node is tree:
        return True

    return isDescendant(tree.left, node) or isDescendant(tree.right, node)


def firstCommonAncestor(tree: BinaryTree, n1: BinaryTree, n2: BinaryTree):
    """
    Design an algorithm and write code to find the first common ancestor
    of two nodes in a binary tree. Avoid storing additional nodes in a data structure. NOTE: This is not
    necessarily a binary search tree.

    I am thinking of approaching it the same way as the linked list problem.
    First go till the upper most node of the tree and calculate the depth of each node.
    Then u cut off excess nodes of the node with greater depth so dat u start at
    same depth. Then u keep going up the parent tree till the first intersection.

    Okay after reading first hint, they saying we mt not have access to the parent node. Well
    den ill check if node is in left side or right side of tree. If they on diff sides, then
    that is first common ancestor.
    """
    if tree is None:
        return tree

    if not isDescendant(tree, n1) or not isDescendant(tree, n2):
        # Check whether nodes actually exist
        return -1

    if n1 is tree or n2 is tree:
        return tree

    isOnLeftn1 = isDescendant(tree.left, n1)
    isOnLeftn2 = isDescendant(tree.left, n2)

    if isOnLeftn2 != isOnLeftn1:
        # Different sides
        return tree

    if isOnLeftn1:
        return firstCommonAncestor(tree.left, n1, n2)

    else:
        return firstCommonAncestor(tree.right, n1, n2)

#4.9
def weaveHelper(result: list, arr1: list, arr2: list):
    """
    Welp somehow u can form all weaves of given two lists by forming weaves of
    arr1[1:] and arr2 and appending it in beg with arr1[0] +
    arr2[1:] and arr1 and appending it in beg with arr2[0]
    :param result:
    :param arr1:
    :param arr2:
    :return:
    """
    if arr1 == [] or arr2 == []:
        return [arr1 + arr2]

    else:
        head1 = arr1[0]
        res1 = weaveHelper(result, arr1[1:], arr2)
        for i in res1:
            i.insert(0, head1)

        head2 = arr2[0]
        res2 = weaveHelper(result, arr1, arr2[1:])
        for i in res2:
            i.insert(0, head2)

        return res1 + res2



def BSTSequences(tree: BinaryTree):
    """
    BST Sequences: A binary search tree was created by traversing through an array from left to right
    and inserting each element. Given a binary search tree with distinct elements, print all possible
    arrays that could have led to this tree.
    :return:

    Ok so the root is always the first item in the list.
    Then the next item is either root.left or it is root.right.
    """
    if tree is None:
        return [[]]

    res = []
    leftLists = BSTSequences(tree.left)
    rightLists = BSTSequences(tree.right)

    for list1 in leftLists:
        for list2 in rightLists:
            weaved = weaveHelper([], list1, list2)
            for i in weaved:
                i.insert(0, tree.root)

            res.extend(weaved)

    return res

# 4.10
def checkSubtree(t1: BinaryTree, t2: BinaryTree):
    """
    Check Subtree: Tl and T2 are two very large binary trees, with Tl much bigger than T2. Create an
    algorithm to determine if T2 is a subtree of Tl.
    A tree T2 is a subtree of Tl if there exists a node n in Tl such that the subtree of n is identical to T2.
    That is, if you cut off the tree at node n, the two trees would be identical.
    :param t1:
    :param t2:
    :return:

    Well one question I have is do they mean subtree by reference or value?
    i.e. Should t2 be an exact same node as in t1? Or the values of t2 should be
    replicated in the subtree of t1.

    One way I can think of when it is based on value is to do a binary tree search
    like in order and see whether the in order of t2 is a subset of t1. So
    u do in order on t1. When u find where t2.root == the node.root u are on currently,
    check whether the two trees equal each other. This wud be O(
    """
    if t2 is None:
        return True

    return checkSubtreeHelper(t1, t2)

def checkSubtreeHelper(t1: BinaryTree, t2: BinaryTree):
    if t1 is None:
        return False

    if isEqual(t1, t2):
        return True
    else:
        return checkSubtreeHelper(t1.left, t2) or checkSubtreeHelper(t1.right, t2)

def isEqual(t1: BinaryTree, t2: BinaryTree):
    """
    Checks whether t1 equals t2
    :param t1:
    :param t2:
    :return:
    """
    if t1 is None and t2 is None:
        return True
    elif t1 is None or t2 is None:
        return False
    else:
        return t1.root == t2.root and isEqual(t1.left, t2.left) and isEqual(t1.right, t2.right)

# 4.12
def pathsWithSum(tree: BinaryTree, reqSum: int):
    """
    So one thing Im thinking of is going through every node and running dfs and
    checking whether u sum up to dat thing. Whenever dat happens, increment a
    counter. Dis implementation works but is a brute force algo. The helper
    costs O(no of nodes in subtree) to run. As every node gets touched once for the
    no of nodes above it. O(n lg n) in balanced and O(n^2) in unbalanced.
    :param tree:
    :return:
    """
    # for every node run paths with sum helper
    ctr = 0
    if tree is not None:
        ctr += pathsWithSumHelper(tree, reqSum)
        ctr += pathsWithSum(tree.left, reqSum)
        ctr += pathsWithSum(tree.right, reqSum)
    return ctr


def pathsWithSumHelper(tree: BinaryTree, reqSum: int):
    """Starting at tree, this func updates ctr to tell how many ways add up to reqsum"""
    ctr = 0
    reqSum = reqSum - tree.root
    if reqSum == 0:
        ctr += 1
    if tree.left:
        ctr += pathsWithSumHelper(tree.left, reqSum)
    if tree.right:
        ctr += pathsWithSumHelper(tree.right, reqSum)

    return ctr

def pathsWithSum2(tree: BinaryTree, reqSum: int):
    cache = {0: 1}

    return pathsWithSumHelper2(tree, reqSum, 0, cache)

def pathsWithSumHelper2(tree: BinaryTree, reqSum: int, currSum: int, sumSet: dict):
    ctr = 0
    if tree is None:
        return ctr


    currSum += tree.root
    prevSum = currSum - reqSum

    ctr += sumSet.get(prevSum, 0)
    sumSet[currSum] = sumSet.get(currSum, 0) + 1

    if tree.left:
        ctr += pathsWithSumHelper2(tree.left, reqSum, currSum, sumSet)

    if tree.right:
        ctr += pathsWithSumHelper2(tree.right, reqSum, currSum, sumSet)

    sumSet[currSum] -= 1

    return ctr

    # if currSum == reqSum:
    #     ctr += 1
    # if tree.left:
    #     ctr += pathsWithSumHelper2(tree.left, reqSum, currSum)
    # if tree.right:
    #     ctr += pathsWithSumHelper2(tree.right, reqSum, currSum)




if __name__ == "__main__":
    a = GraphDirected()
    a.addVertex(1)
    a.addVertex(2)
    a.addVertex(3)
    a.addVertex(4)
    a.addVertex(5)
    a.addEdge(1, 2)
    a.addEdge(2, 3)
    a.addEdge(1, 4)
    a.addEdge(5, 3)

    binaryTree = BinaryTree(10)
    # binaryTree.left = BinaryTree(1)
    # binaryTree.right = BinaryTree(2)
    binaryTree.left = BinaryTree(5)
    binaryTree.right = BinaryTree(-3)
    binaryTree.left.left = BinaryTree(3)
    binaryTree.left.right = BinaryTree(2)
    binaryTree.left.right.right = BinaryTree(1)
    binaryTree.right.right = BinaryTree(11)
    binaryTree.left.left.left = BinaryTree(3)
    binaryTree.left.left.right= BinaryTree(-2)

    binaryTreePar = BinaryTreePar(1)

    binaryTree2 = BinaryTree(1)
    binaryTree2.left = BinaryTree(-2)
    binaryTree2.right = BinaryTree(-3)



