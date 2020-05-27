#pragma once

#include "typedefs.h"
#include "hash_table.h"

namespace HashDagUtils
{
    HOST_DEVICE_RECURSIVE uint32 count_children(HashTable& hashTable, uint32 level, uint32 levels, uint32 index)
    {
        const uint32* nodePtr = hashTable.get_sys_ptr(level, index);
        if (level == levels - 2)
        {
            return Utils::popc(nodePtr[0]) + Utils::popc(nodePtr[1]);
        }
        else
        {
            uint32 count = 0;
            for (uint32 child = 1; child < Utils::total_size(nodePtr[0]); child++)
            {
                count += count_children(hashTable, level + 1, levels, nodePtr[child]);
            }
            checkEqual(count, nodePtr[0] >> 8u);
            return count;
        }
    }
}