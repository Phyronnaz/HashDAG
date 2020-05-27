from tools import get_array, dags, results_prompt, defines

results_prompt("bucket_count_memory_overhead")

for i in range(len(dags)):
    data = dags[i]
    virtual_size = get_array(data, "virtual_size")
    allocated_size = get_array(data, "allocated_size")
    print("buckets bits: {}; used: {}; allocated: {}; wasted: {}".format(
        defines[i]["BUCKETS_BITS_FOR_LOW_LEVELS"],
        virtual_size[0],
        allocated_size[0],
        allocated_size[0] - virtual_size[0]))
