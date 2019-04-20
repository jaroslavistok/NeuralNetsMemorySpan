class LCS:
    def calc_cache_pos(self, strings, indexes):
        factor = 1
        pos = 0
        for s, i in zip(strings, indexes):
            pos += i * factor
            factor *= len(s)
        return pos

    def lcs_back(self, strings, indexes, cache):
        if -1 in indexes:
            return ""
        match = all(strings[0][indexes[0]] == s[i]
                    for s, i in zip(strings, indexes))
        if match:
            new_indexes = [i - 1 for i in indexes]
            result = self.lcs_back(strings, new_indexes, cache) + strings[0][indexes[0]]
        else:
            substrings = [""] * len(strings)
            for n in range(len(strings)):
                if indexes[n] > 0:
                    new_indexes = indexes[:]
                    new_indexes[n] -= 1
                    cache_pos = self.calc_cache_pos(strings, new_indexes)
                    if cache[cache_pos] is None:
                        substrings[n] = self.lcs_back(strings, new_indexes, cache)
                    else:
                        substrings[n] = cache[cache_pos]
            result = max(substrings, key=len)
        cache[self.calc_cache_pos(strings, indexes)] = result
        return result

    def lcs(self, strings):
        if len(strings) == 0:
            return ""
        elif len(strings) == 1:
            return strings[0]
        else:
            cache_size = 1
            for s in strings:
                cache_size *= len(s)
                print(cache_size)
            cache = [None] * cache_size
            indexes = [len(s) - 1 for s in strings]
            return self.lcs_back(strings, indexes, cache)

    def lcs_length(self, strings):
        return len(self.lcs(strings))