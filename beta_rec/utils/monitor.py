import os
import time
from threading import Thread

import cpuinfo
import GPUtil
import psutil
from tensorboardX import SummaryWriter


class Monitor(Thread):
    """Monitor Class."""

    def __init__(self, log_dir, delay=1, gpu_id=0, verbose=False):
        """Initialize monitor, log_dir and gpu_id are needed."""
        super(Monitor, self).__init__()

        DEVICE_ID_LIST = GPUtil.getAvailable(
            order="memory", limit=1
        )  # get the fist gpu with the lowest load
        if len(DEVICE_ID_LIST) < 1 or gpu_id is None:
            self.hasgpu = False
        else:
            self.hasgpu = True

        self.gpu_id = gpu_id
        self.start_time = time.time()  # Start time
        self.verbose = verbose  # if update the usage status during the process
        self.stopped = False  # flag for stop the monitor
        self.delay = delay  # Time between calls to GPUtil
        self.pid = os.getpid()
        self.writer = SummaryWriter(log_dir=log_dir)  # tensorboard writer
        label = "brand"
        if "brand_raw" in cpuinfo.get_cpu_info().keys():
            label = "brand_raw"
        self.writer.add_text(
            "device/CPU",
            "cpu count: {:d} \t brand: {:s}".format(
                os.cpu_count(), cpuinfo.get_cpu_info()[label]
            ),
            0,
        )
        self.writer.add_text(
            "device/RAM",
            "Current RAM - total:\t {:.3f}GB;".format(
                psutil.virtual_memory().total / 2.0 ** 30
            ),
            0,
        )
        self.count = 0  # Count for calculate the average usage

        self.GPU_memoryUsed = []
        self.GPU_memoryFree = []
        self.CPU_load = []
        self.memoryUsed = []

        if self.hasgpu:
            self.GPU = GPUtil.getGPUs()[self.gpu_id]
            self.GPU_memoryTotal = (
                self.GPU.memoryTotal / 2.0 ** 10
            )  # Total gpu memory amount in GB
            self.writer.add_text(
                "device/GPU",
                "Current GPU (ID:{:d}) name:{:s} ".format(self.gpu_id, self.GPU.name)
                + "Total_GPU_memory: {:.3f}GB;".format(self.GPU_memoryTotal),
                0,
            )

        if verbose:
            devices_status()
        self.start()

    def write_cpu_status(self):
        """Write CPU status."""
        CPU_load = psutil.Process(self.pid).cpu_percent(interval=1)
        self.writer.add_scalars(
            "device/cpu",
            {"CPU_load (%)": CPU_load},
            self.count,
        )
        self.CPU_load.append(CPU_load)

    def write_mem_status(self):
        """Write memory usage status."""
        memoryUsed = (
            psutil.Process(self.pid).memory_info()[0] / 2.0 ** 30
        )  # current app memory use in GB
        self.writer.add_scalars(
            "device/mem",
            {"memory_used (GB)": memoryUsed},
            self.count,
        )
        self.memoryUsed.append(memoryUsed)

    def write_gpu_status(self):
        """Write gpu usage status."""
        self.GPU = GPUtil.getGPUs()[self.gpu_id]
        GPU_load = self.GPU.load * 100
        GPU_memoryUsed = self.GPU.memoryUsed / self.GPU_memoryTotal * 100
        GPU_memoryFree = self.GPU.memoryFree / self.GPU_memoryTotal * 100
        self.writer.add_scalars(
            "device/GPU",
            {
                "GPU_load (%)": GPU_load,
                "GPU_memory_used (%)": GPU_memoryUsed,
                "GPU_memory_free (%)": GPU_memoryFree,
            },
            self.count,
        )
        self.GPU_memoryUsed.append(GPU_memoryUsed)
        self.GPU_memoryFree.append(GPU_memoryFree)

    def run(self):
        """Run the monitor."""
        while not self.stopped:
            self.count += 1
            self.write_cpu_status()
            self.write_mem_status()
            if self.hasgpu:
                self.write_gpu_status()

    def stop(self):
        """Stop the monitor."""
        self.run_time = time.time() - self.start_time
        print("Program running time:%d seconds" % self.run_time)
        self.stopped = True
        return self.run_time


def print_gpu_stat(gpu_id=None):
    """Print GPU status."""
    if gpu_id is None:
        gpu_ids = GPUtil.getAvailable(limit=10)
        for gpu_id in gpu_ids:
            GPU = GPUtil.getGPUs()[gpu_id]
            GPU_load = GPU.load * 100
            GPU_memoryUtil = GPU.memoryUtil / 2.0 ** 10
            GPU_memoryTotal = GPU.memoryTotal / 2.0 ** 10
            GPU_memoryUsed = GPU.memoryUsed / 2.0 ** 10
            GPU_memoryFree = GPU.memoryFree / 2.0 ** 10
            print("Current GPU (ID:{:d}) name:\t{:s}".format(gpu_id, GPU.name))
            print("Total_GPU_memory:\t{:.3f}GB;".format(GPU_memoryTotal))
            print("GPU_memoryUtil:\t{:.3f}GB;".format(GPU_memoryUtil))
            print("GPU_memoryUsed:\t{:.3f}GB;".format(GPU_memoryUsed))
            print("GPU_memoryFree:\t{:.3f}GB;".format(GPU_memoryFree))
            print("GPU_load:\t{:.3f}GB;".format(GPU_load))
    else:
        GPU = GPUtil.getGPUs()[gpu_id]
        GPU_load = GPU.load * 100
        GPU_memoryUtil = GPU.memoryUtil / 2.0 ** 10
        GPU_memoryTotal = GPU.memoryTotal / 2.0 ** 10
        GPU_memoryUsed = GPU.memoryUsed / 2.0 ** 10
        GPU_memoryFree = GPU.memoryFree / 2.0 ** 10
        print("Current GPU (ID:{:d}) name:{:s}".format(gpu_id, GPU.name))
        print("Total_GPU_memory: {:.3f}GB;".format(GPU_memoryTotal))
        print("GPU_memoryUsed:{:.3f}GB;".format(GPU_memoryUsed))
        print("GPU_memoryFree:{:.3f}GB;".format(GPU_memoryFree))
        print("GPU_load:{:.3f}GB;".format(GPU_load))


"""
some static methods
"""


def print_cpu_stat():
    """Print CPU status."""
    label = "brand"
    if "brand_raw" in cpuinfo.get_cpu_info().keys():
        label = "brand_raw"
    print(
        "Cpu count: {:d} \t brand: {:s}".format(
            os.cpu_count(), cpuinfo.get_cpu_info()[label]
        )
    )
    print("Avg_load_1m: \t{:.3f}%%;".format(os.getloadavg()[0]))
    print("Avg_load_5m:\t{:.3f}%%;".format(os.getloadavg()[1]))
    print("Avg_load_15m:\t{:.3f}%%;".format(os.getloadavg()[2]))


def print_mem_stat(memoryInfo=None):
    """Print memory status."""
    # Main memory info
    if memoryInfo is None:
        memoryInfo = (
            psutil.virtual_memory()
        )  # svmem(total, available, percent, used, free, active, inactive, buffers, cached, shared, slab)
    print("Current RAM - total:\t {:.3f}GB;".format(memoryInfo.total / 2.0 ** 30))
    print(
        "Current RAM - available:\t{:.3f}GB;".format(memoryInfo.available / 2.0 ** 30)
    )
    print("Current RAM - used:\t{:.3f}GB;".format(memoryInfo.used / 2.0 ** 30))
    print("Current RAM - free:\t{:.3f}GB;".format(memoryInfo.free / 2.0 ** 30))


# print current devices available
def devices_status():
    """Print current devices status."""
    print_cpu_stat()
    print_mem_stat()
    print_gpu_stat()
