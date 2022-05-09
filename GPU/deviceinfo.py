#!/usr/bin/env python
# File: deviceinfo.py
# Name: D.Saravanan
# Date: 09/05/2022

""" Script to print out some information about the OpenCL devices and platforms available on your system """

import pyopencl as cl

# create a list of all the platform IDs
platforms = cl.get_platforms()

print("\nNumber of OpenCL platforms:", len(platforms))

print("\n----------------------------------------")

# investigate each platform
for p in platforms:
    # print out some information about the platform
    print("Platform:" , p.name)
    print("Vendor:" , p.vendor)
    print("Version:" , p.version)

    # discover all devices
    devices = p.get_devices()
    print("Number of devices:" , len(devices))

    # investigate each device
    for d in devices:
        print("\t----------------------------------------")
        # print out some information about the devices
        print("\t\tName:" , d.name)
        print("\t\tVersion:", d.opencl_c_version)
        print("\t\tMax. Compute Units:" , d.max_compute_units)
        print("\t\tLocal Memory Size:" , d.local_mem_size/1024, "KB")
        print("\t\tGlobal Memory Size:" , d.global_mem_size/(1024*1024), "MB")
        print("\t\tMax Alloc Size:" , d.max_mem_alloc_size/(1024*1024), "MB")
        print("\t\tMax Work-group Total Size:" , d.max_work_group_size)
        print("\t\tCache Size:" , d.global_mem_cacheline_size)

        # find the maximum dimensions of the work-groups
        dim = d.max_work_item_sizes
        print("\t\tMax Work-group Dims:(" , dim[0], " ".join(map(str, dim[1:])), ")")
        
        print("\t----------------------------------------")

    print("\n----------------------------------------")
