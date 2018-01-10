# from table import Table, Ball

class Table:
	def __init__(self, list_of_coords):
		self.list_of_coords = list_of_coords

	def set_coords(self, list_of_coords):
		self.list_of_coords = list_of_coords
		
	def get_coords(self):
		return self.list_of_coords