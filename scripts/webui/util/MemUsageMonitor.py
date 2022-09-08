from util.imports import *

class MemUsageMonitor(threading.Thread):
	stop_flag = False
	max_usage = 0
	total = -1

	def __init__(self, name):
		threading.Thread.__init__(self)
		self.name = name

	def run(self):
		try:
			pynvml.nvmlInit()
		except:
			print(f"[{self.name}] Unable to initialize NVIDIA management. No memory stats. \n")
			return
		print(f"[{self.name}] Recording max memory usage...\n")
		handle = pynvml.nvmlDeviceGetHandleByIndex(defaults.general.gpu)
		self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
		while not self.stop_flag:
			m = pynvml.nvmlDeviceGetMemoryInfo(handle)
			self.max_usage = max(self.max_usage, m.used)
			# print(self.max_usage)
			time.sleep(0.1)
		print(f"[{self.name}] Stopped recording.\n")
		pynvml.nvmlShutdown()

	def read(self):
		return self.max_usage, self.total

	def stop(self):
		self.stop_flag = True

	def read_and_stop(self):
		self.stop_flag = True
		return self.max_usage, self.total
