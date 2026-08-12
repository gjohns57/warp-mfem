from math import ceil
from typing import Any

import warp as wp
from sympy import nextprime


@wp.func
def hash64(key: wp.uint64):
    fnv_prime = wp.uint64(0x00000100000001B3)
    hash = wp.uint64(0xCBF29CE484222325)

    for i in range(8):
        hash = wp.bit_xor(
            hash, wp.bit_and(wp.rshift(key, wp.uint64(8 * i)), wp.uint64(0xFF))
        )
        hash *= fnv_prime

    return hash


@wp.func
def pack_tri(tri: wp.vec3i) -> wp.uint64:
    return wp.bit_or(
        wp.lshift(wp.bit_and(wp.uint64(tri[0]), wp.uint64(0x1FFFFF)), wp.uint64(42)),
        wp.bit_or(
            wp.lshift(
                wp.bit_and(wp.uint64(tri[1]), wp.uint64(0x1FFFFF)), wp.uint64(21)
            ),
            wp.bit_and(wp.uint64(tri[2]), wp.uint64(0x1FFFFF)),
        ),
    )


@wp.func
def hashtable_insert(
    keys: wp.array[wp.uint64],
    size: wp.int32,
    key: wp.uint64,
    empty: wp.uint64,
) -> tuple[wp.int32, wp.bool]:
    index = wp.int32(hash64(key) % wp.uint64(size))
    i = wp.int32(0)

    test = wp.atomic_cas(keys, (index + i * i) % size, empty, key)
    while test != empty and test != key:
        i += 1
        test = wp.atomic_cas(keys, (index + i * i) % size, empty, key)

    return (index + i * i, test == empty)


@wp.func
def hashtable_find(
    keys: wp.array[wp.uint64],
    size: wp.int32,
    key: wp.uint64,
    empty: wp.uint64,
) -> wp.int32:
    index = wp.int32(hash64(key) % wp.uint64(size))
    i = wp.int32(0)

    test = keys[(index + i * i) % size]
    while test != key and test != empty:
        i += 1
        test = keys[(index + i * i) % size]

    return (index + i * i) % size


@wp.func
def hashmap_insert_edge(
    keys: wp.array[wp.uint64],
    values: wp.array[Any],
    hashmap_size: wp.int32,
    edge: wp.vec2i,
    value: Any,
):
    key = wp.cast(edge, wp.uint64)
    empty = wp.cast(wp.vec2i(), wp.uint64)
    index, was_inserted = hashtable_insert(keys, hashmap_size, key, empty)
    values[index] = value

    return was_inserted


@wp.func
def hashmap_find_edge(
    keys: wp.array[wp.uint64],
    values: wp.array[wp.int32],
    size: wp.int32,
    edge: wp.vec2i,
) -> wp.int32:
    """Find whether a triangle is in the hashset"""
    key = wp.cast(edge, wp.uint64)
    empty = wp.cast(wp.vec2i(), wp.uint64)

    index = hashtable_find(keys, size, key, empty)
    if keys[index] != empty:
        return values[index]
    else:
        return -1


@wp.func
def hashmap_remove_edge(
    keys: wp.array[wp.uint64],
    values: wp.array[wp.int32],
    size: wp.int32,
    edge: wp.vec2i,
) -> wp.bool:
    """Remove a triangle from the hashset"""
    key = wp.cast(edge, wp.uint64)
    empty = wp.cast(wp.vec2i(), wp.uint64)

    index = hashtable_find(keys, size, key, empty)
    if keys[index] != empty:
        keys[index] = empty
        values[index] = -1
        return True
    else:
        return False


@wp.func
def hashbag_add_tri(
    keys: wp.array[wp.uint64],
    counts: wp.array[wp.int32],
    size: wp.int32,
    tri: wp.vec3i,
) -> wp.int32:
    """Insert a triangle to the bag (multiset) and return the new old count"""
    key = pack_tri(tri)
    empty = pack_tri(wp.vec3i())

    index, _is_new_entry = hashtable_insert(keys, size, key, empty)
    return wp.atomic_add(counts, index, 1)


@wp.func
def hashset_insert_edge(
    keys: wp.array[wp.uint64],
    size: wp.int32,
    edge: wp.vec2i,
) -> wp.bool:
    key = wp.cast(edge, dtype=wp.uint64)
    empty = wp.cast(wp.vec2i(), dtype=wp.uint64)

    _index, is_new_entry = hashtable_insert(keys, size, key, empty)
    return is_new_entry


@wp.func
def hashbag_find_tri(
    keys: wp.array[wp.uint64],
    counts: wp.array[wp.int32],
    size: wp.int32,
    tri: wp.vec3i,
) -> wp.int32:

    key = pack_tri(tri)
    empty = pack_tri(wp.vec3i())

    index = hashtable_find(keys, size, key, empty)

    if keys[index] == empty:
        return 0
    else:
        return counts[index]


@wp.func
def hashset_find_edge(
    keys: wp.array[wp.uint64],
    size: wp.int32,
    edge: wp.vec2i,
) -> wp.bool:

    key = wp.cast(edge, wp.uint64)
    empty = wp.cast(wp.vec2i(), wp.uint64)

    index = hashtable_find(keys, size, key, empty)
    return keys[index] == key


def get_hashtable_size(estimated_load: int, load_factor: float) -> int:
    return nextprime(ceil(estimated_load / load_factor))
