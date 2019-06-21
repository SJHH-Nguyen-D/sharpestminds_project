'''Binary search trees are a common problem in coding interviews
This is why I will get familiar with learning about them and solving them
The tree should be structured to code like this:


User
bst = Tree()
bst.insert(14)
bst.preorder() # tree traversal function
bst.postorder() # tree traversal function
bst.inorder() # tree traversal function
'''
class Node:
	""" Node helper class of the Tree Class """
	def __init__(self, value):
		self.value = value
		self.left = None
		self.right = None

	def insert(self, data):
		""" Inserts the value into the node """
		# Don't want duplicates in our tree. Return False if data is duplicate of 
		# Pre-existing value
		if self.value == data:
			return False
		elif self.value > data:
			# we want to check if there is a left child already
			if self.left:
				# if there is one, insert the data into it
				return self.left.insert(data)
			# if there is no left child, it will create a new node using that data as its value
			else:
				self.left = Node(data)
				return True
		else:
			if self.right:
				return self.right.insert(data)
			else:
				self.right = Node(data)
				return True

	def find(self, data):
		""" Finds a node given a value
			Most of the heavy lifting for the find function is the Node class.
		"""
		if (self.value == data):
			return True
		elif self.value > data:
			if self.left:
				return self.left.find(data)
			else:
				return False
		else:
			if self.right:
				return self.right.find(data)
			else:
				return False

	def preorder(self):
		""" Preorder traversal function from the Node class """
		if self:
			print(str(self.value))
			if self.left:
				self.left.preorder()
			if self.right:
				self.right.preorder()

	def postorder(self):
		""" Postorder traversal function from the Node class """
		if self:
			print(str(self.value))
			if self.left:
				self.left.postorder()
			if self.right:
				self.right.postorder()

	def inorder(self):
		""" InOrder traversal function from the Node class """
		if self:
			print(str(self.value))
			if self.left:
				self.left.inorder()
			if self.right:
				self.right.inorder()
			print(str(self.value))


class Tree:
	""" Binary Search Tree class"""
	def __init__(self):
		self.root = None

	def insert(self, data):
		""" Inserts a node into the BST """
		if self.root:
			return self.root.insert(data)
		else:
			self.root = Node(data)
			return True

	def find(self, data):
		""" Finds the value of a node in the Tree
			The find function in the Tree class acts as more of an interface for the user.
		"""
		if self.root:
			self.root.find(data)
		else:
			return False

	def preorder(self):
		""" Preorder traversal function of the BST """
		print("PreOrder")
		self.root.preorder()

	def postorder(self):
		""" postorder traversal function of the BST """
		print("PostOrder")
		self.root.postorder()

	def inorder(self):
		""" Inorder traversal function of the BST """
		print("InOrder")
		self.root.inorder()


########### Instantiating the Tree and inserting nodes into it ##################

bst = Tree()
bst.insert(10) # returns True or False if the value was inserted successfulling into the tree
print(bst.insert(20)) # returns True or False if the value was inserted successfulling into the tree
print(bst.insert(2))

values = [10, 3, 2, 10, 5, 7, 200, 20, 100]
for val in values:
	print(bst.insert(val))

bst.preorder() # traversal of the tree
bst.postorder() # postorder traversal of the tree
bst.inorder() # inorder traversal of the tree