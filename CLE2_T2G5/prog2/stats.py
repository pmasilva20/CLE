import math
dict_stats_128={
    "mat128_32_1" :[],
    "mat128_32_2" :[],
    "mat128_32_4" :[],
    "mat128_32_8" :[],

    "mat128_64_1" :[],
    "mat128_64_2" :[],
    "mat128_64_4" :[],
    "mat128_64_8" :[],
    
    "mat128_128_1" :[],
    "mat128_128_2" :[],
    "mat128_128_4" :[],
    "mat128_128_8" :[],
    
    "mat128_256_1" :[],
    "mat128_256_2" :[],
    "mat128_256_4" :[],
    "mat128_256_8" :[],
}

dict_stats_512={
    "mat512_32_1" :[],
    "mat512_32_2" :[],
    "mat512_32_4" :[],
    "mat512_32_8" :[],

    "mat512_64_1" :[],
    "mat512_64_2" :[],
    "mat512_64_4" :[],
    "mat512_64_8" :[],
    
    "mat512_128_1" :[],
    "mat512_128_2" :[],
    "mat512_128_4" :[],
    "mat512_128_8" :[],
    
    "mat512_256_1" :[],
    "mat512_256_2" :[],
    "mat512_256_4" :[],
    "mat512_256_8" :[],
}

#TODO: Para cada stat calcular variância ou desvio padrão or both Depois meter numa tabela  slides spreedshit

print(dict_stats_128)

for values in dict_stats_128:
    mean=sum(values)/len(values)
    variance=sum([(x - mean)*(x - mean) for x in values])/len(values)
    maxdev=max([abs(x - mean) for x in values])
    mad=sum([abs(x - mean) for x in values])/len(values)
    stddev=math.sqrt(variance)

    