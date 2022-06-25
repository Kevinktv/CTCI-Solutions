"""
Solutions to the Linked List problems CTCI
"""
from __future__ import annotations
from typing import Optional


class Node:
    item: int
    next: Optional[Node]

    def __init__(self, item: int):
        self.item = item
        self.next = None


l1 = Node(1)
l1.next = Node(2)
l1.next.next = Node(3)
l1.next.next.next = Node(4)
temp = Node(5)
temp.next = Node(6)
temp.next.next = Node(7)
temp.next.next.next = temp
l1.next.next.next.next = temp

l2 = temp




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
        print(head.item, " ")
        head = head.next


class RemoveDups:
    """
    Write code to remove duplicates from an unsorted linked list.
    FOLLOW UP
    How would you solve this problem if a temporary buffer is not allowed?


    If buffer is allowed, what we can do is simply just add every element u see in the
    linked list to a hash set and den only add stuff that have not appeared yet to the linked
    list. This is O(n) time and O(n) space.


    """

    def removeDupsWBuffer(self, head: Node):

        table = set()
        prev = None
        curr = head

        while curr is not None:
            if curr.item not in table:
                table.add(curr.item)
                prev = curr
            else:
                prev.next = curr.next

            curr = curr.next

    def removeDupsWithoutBuff(self, head: Node):
        """Removes duplicates without a buffer. For every item, go through list to
        see whether it occurs again. If so remove them from list.
        O(n^2)"""

        curr = head

        while curr is not None:
            temp = curr.next
            tempPrev = curr
            while temp is not None:
                if temp.item == curr.item:  # Delete da node
                    tempPrev.next = temp.next
                else:
                    tempPrev = temp

                temp = temp.next

            curr = curr.next


class DeleteFromMiddle:
    """
    Implement an algorithm to delete a node in the middle (i.e., any node but
    the first and last node, not necessarily the exact middle) of a singly linked list, given only access to
    that node.
    EXAMPLE
    lnput:the node c from the linked list a->b->c->d->e->f
    Result: nothing is returned, but the new linked list looks like a->b->d->e- >f
    """

    def deleteFromMiddle(self, node: Node):
        """
        We do not have access to the head.
        To delete node, we can move data from next node into node and then delete next node
        Also as node in middle we gud.
        :param node: The node which needs to be deleted
        :return: None
        """
        nextnode = node.next
        node.item = nextnode.item
        node.next = nextnode.next


class ReturnKthToLast:
    """
    Implement an algorithm to find the kth to last element of a singly linked list.

    So when k = 1 delete last node. So k != 0

    One implementation that I can think of is first find the length of the list.
    So if list is 1 - 2 - 3 - 4 - 5, length is 5. This takes one iteration.
    Now if we need to delete the kth to last element, we delete len(lst) - k item.
    NOTE: len - k item assumes a 0 based indexing. So after that many iter, we reached elem
    to delete.
    This algo is O(n) but requires two passes through the linked list.
    """

    def returnKthToLastLen(self, head: Node, k: int):
        """
        Implementation using the lenght method.
        :param k: The thingy to delete
        :param head: The head
        :return:
        """
        length = 0

        curr = head
        while curr is not None:
            length += 1
            curr = curr.next

        indexToDelete = length - k  # After how many iter to delete
        iterNo = 0

        curr = head

        if indexToDelete < 0:  # Ohoh den u entered wrong k
            return -999

        while curr is not None:
            if iterNo == indexToDelete:
                return curr.item
            iterNo += 1
            curr = curr.next

    def returnKthToLastIterative(self, head: Node, k: int):
        """
        Iterative implem of it. So basicallly, if u want the kth to last item, u have a
        second pointer trailing behind the "curr" by k - 1 units. So if k = 1, then
        second pointer and first travel together.
        :param head:
        :return:
        """
        curr = head
        trail = None
        ctr = 1
        while curr is not None:
            if trail is not None:
                trail = trail.next

            if ctr == k:
                trail = head

            curr = curr.next

            ctr += 1

        return trail.item


class Partition:
    """
    Write code to partition a linked list around a value x, such that all nodes less than x come
    before all nodes greater than or equal to x. If x is contained within the list, the values of x only need
    to be after the elements less than x (see below). The partition element x can appear anywhere in the
    "right partition"; it does not need to appear between the left and right partitions.
    """

    def partition(self, head: Node, x: int):
        """

        """
        smallerHead = None
        biggerHead = None

        smallerCurr = None
        biggerCurr = None

        curr = head

        while curr is not None:
            node = Node(curr.item)
            if curr.item < x:
                if smallerHead is None:  # First item is added
                    smallerHead = node
                    smallerCurr = smallerHead
                else:
                    smallerCurr.next = node
                    smallerCurr = node

            else:
                if biggerHead is None:  # First item is added
                    biggerHead = node
                    biggerCurr = biggerHead
                else:
                    biggerCurr.next = node
                    biggerCurr = node

            curr = curr.next

        if smallerHead is None:  # Nothing smaller than parti
            return biggerHead

        else:
            smallerCurr.next = biggerHead
            return smallerHead


class SumList:
    """
    Sum Lists: You have two numbers represented by a linked list, where each node contains a single
    digit. The digits are stored in reverse order, such that the 1 's digit is at the head of the list. Write a
    function that adds the two numbers and returns the sum as a linked list.
    EXAMPLE
    Input: (7-> 1 -> 6) + (5 -> 9 -> 2).That is,617 + 295.
    Output: 2 -> 1 -> 9. That is, 912.
    FOLLOW UP
    Suppose the digits are stored in forward order. Repeat the above problem.
    EXAMPLE
    lnput:(6 -> 1 -> 7) + (2 -> 9 -> 5).That is,617 + 295.
    Output: 9 - > 1 -> 2. That is, 912.


    One approach I can straight up think of is getting the two numbers individually
    by going through the LL and then adding dem up normally and converts it back to a LL. It is
    O(n) in time and space but I dont think that is how they want us to do it.

    Another approach for the first one I think of is go thorugh the linked list nodes in
    both simult. And add htose two and keep track of excess which is added to next step.
    """

    def sumList1(self, l1: Node, l2: Node):
        """
        Does the adding approach by going through both lists simultaneously and adding
        digit by digit with carry. Assuming that both lists have atleast one digit in it.
        :param l1:
        :param l2:
        :return:
        """

        h1 = l1
        h2 = l2
        new = None
        temp = None
        rem = 0

        while h1 is not None or h2 is not None:

            h1dig = h1.item if h1 is not None else 0
            h2dig = h2.item if h2 is not None else 0

            res = h1dig + h2dig + rem
            dig = res % 10
            rem = res // 10

            if new is None:
                new = Node(dig)
                temp = new
            else:
                temp.next = Node(dig)
                temp = temp.next

            if h1 is not None:
                h1 = h1.next
            if h2 is not None:
                h2 = h2.next

        if rem != 0:
            temp.next = Node(rem)

        return new


class Palindrome:
    """
    Implement a function to check if a linked list is a palindrome.
    """

    def palindrome1(self, head: Node):
        """
        This checks whether the linked list is the same in reverse using a stack. It goes
        through entire ll which aint req
        :param head:
        :return:
        """
        curr = head
        stack = []
        while curr is not None:
            stack.append(curr.item)
            curr = curr.next

        curr = head
        while curr is not None:
            item = stack.pop()
            if curr.item != item:
                return False
            curr = curr.next
        return True

    def palindrome2(self, head: Node):
        """
        This one only stores half of the LL into the stack and checks remaining half if
        they equal
        :param head:
        :return:
        """
        slow = head
        fast = head
        stack = []

        while fast is not None and fast.next is not None:
            stack.append(slow.item)
            slow = slow.next
            fast = fast.next.next

        slow = slow if fast is None else slow.next
        while slow is not None:
            item = stack.pop()
            if slow.item != item:
                return False
            slow = slow.next
        return True


class Intersection:
    """
    Intersection: Given two (singly) linked lists, determine if the two lists intersect. Return the intersecting
    node. Note that the intersection is defined based on reference, not value. That is, if the kth
    node of the first linked list is the exact same node (by reference) as the jth node of the second
    linked list, then they are intersecting.


    One bruteforce implementation is for every node is l1 check whether it refers to a node in l2.
    O(n^2).

    Another method I can thinnk of is by hashing the addresses. So go through LL 1 and hash em by
    their ids. Then go through LL 2 and check whether those nodes are present in the hash table. If
    atleast one node like dat exists, we are done. Dis be O(m + n) time and space.

    An O(m + n) but constant space approach is given below.
    """
    def intersection(self, h1: Node, h2: Node):
        c1 = h1
        c2 = h2

        len1 = 1
        len2 = 1

        while c1.next is not None:
            len1 += 1
            c1 = c1.next

        while c2.next is not None:
            len2 += 1
            c2 = c2.next

        if c1 is not c2:
            print("bruh")
            # Then the last node is different so they cannot be itersecting
            return None


        # Now we gotta find where they intersect. So I will go through both LL simult
        # starting from the same length point. And check where the nodes equate first

        c1 = h1
        c2 = h2

        while c1 is not None and c2 is not None:
            if len1 > len2:
                c1 = c1.next
                len1 -= 1
            elif len1 < len2:
                c2 = c2.next
                len2 -= 1
            else:
                if c1 is c2:
                    return c1

                c1 = c1.next
                c2 = c2.next


class LoopDetection:
    """
    Loop Detection: Given a circular linked list, implement an algorithm that returns the node at the
    beginning of the loop.
    DEFINITION
    Circular linked list: A (corrupt) linked list in which a node's next pointer points to an earlier node, so
    as to make a loop in the linked list.
    EXAMPLE
    Input: A -> B -> C - > D -> E -> C [the same C as earlier]
    Output: C


    Again the first approach I can think of is by hashing. Go through LL and hash the ids.
    Check when the first time the search succeeds, that is the Node we need. This is O(n) space
    and time. (Kinda similar to graph cycle detection by coloring nodes)

    So a cycle exists if two poiters travelling at different speeds intersect with each other.
    Also note that once they intersect, the distance between the start node and beginning of
    cycle = distance between intersection to start node.


    """
    def loopDetection(self, head: Node):
        slow = head
        fast = head
        firstiter = True

        while not(fast is None or fast.next is None or (slow == fast and not firstiter)):
            slow = slow.next
            fast = fast.next.next
            if firstiter:
                firstiter = False

        if fast is None or fast.next is None:
            return None

        # Intersects
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next

        return slow

