# 上海海事大学
# 软件工程 王彪
# 开发时间： 2025/3/29 19:47
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insertNode(self, data, position):
        newNode = Node(data)
        if self.head is None or position == 0:
            newNode.next = self.head
            self.head = newNode
            return
        current = self.head
        count = 0
        while current.next and count < position - 1:
            current = current.next
            count += 1
        newNode.next = current.next
        current.next = newNode

    def deleteNode(self, value):
        if self.head is None:
            return
        if self.head.data == value:
            self.head = self.head.next
            return
        current = self.head
        while current.next and current.next.data != value:
            current = current.next
        if current.next:
            current.next = current.next.next

    def findNode(self, value):
        current = self.head
        while current:
            if current.data == value:
                return True
            current = current.next
        return False

    def modifyNode(self, position, newData):
        current = self.head
        count = 0
        while current and count < position:
            current = current.next
            count += 1
        if current is None:
            raise IndexError("位置无效")
        current.data = newData

    def printList(self):
        current = self.head
        while current:
            print(current.data, end=" ")
            current = current.next
        print()

# 测试代码
llist = LinkedList()
llist.insertNode(1, 0)
llist.insertNode(2, 1)
llist.insertNode(3, 2)
llist.printList()
llist.deleteNode(2)
llist.printList()
print("查找 3:", llist.findNode(3))
try:
    llist.modifyNode(1, 4)
except IndexError as e:
    print(e)
llist.printList()