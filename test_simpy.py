from time import perf_counter
def calc(s):
    seen = set()
    for ch in s:
        seen.add(ch)

        if len(seen) == 26:
            break
    else:
        return False
    
    return True

start = perf_counter()
res = calc("qdcarlwydorfpvsjahyeojeuaywqgvlnzxggnthhljqgzjeoozmurjakxqgzbfqdyhnqrfqbvhmqpsprwaltkvwuecmzvmrlzrimqdcarlwydorfpvsjahyeojeuaywqgvlnzxggnthhljqgzjeoozmurjakxqgzbfqdyhnqrfqbvhmqpsprwaltkvwuecmzvmrlzrim")
print(res, (perf_counter() - start)*1000)
